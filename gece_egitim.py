# -*- coding: utf-8 -*-
"""
gece_egitim.py — Gece Boyunca Headless Eğitim
=============================================
- Görsel KAPALI → maksimum hız (5-10x)
- Her 25 bölümde bir otomatik kayıt
- En iyi skoru takip eder, en iyi modeli ayrı saklar
- Saat bazlı log → sabah analiz edilebilir
- Ctrl+C ile güvenli durdurma

KULLANIM:
    python gece_egitim.py
    python gece_egitim.py --model model_bolum50_odul16584.pth
"""
import sys, os, glob, time, math, argparse
import numpy as np
import torch

# Türkçe karakter desteği
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── AYARLAR ──────────────────────────────────────────────────
BOLUM_SAYISI     = 2000      # ~6-8 saatlik eğitim (headless)
KAYIT_ARASI      = 25        # Her 25 bölümde kaydet (güvenli)
EN_IYI_KAYIT     = True      # En iyi skoru ayrı sakla
LOG_DOSYASI      = "gece_egitim_log.txt"
BASLANGIC_MODEL  = None      # argparse ile doldurulur
MAX_ADIM_BOLUM   = 2000      # Bölüm başına maksimum adım (daha hızlı döngü)
# ─────────────────────────────────────────────────────────────


def sure_format(saniye):
    """Saniyeyi okunabilir formata çevirir."""
    s = int(saniye)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def log_yaz(mesaj, dosya=LOG_DOSYASI):
    """Hem terminale hem log dosyasına yazar."""
    zaman = time.strftime("%H:%M:%S")
    satir = f"[{zaman}] {mesaj}"
    print(satir, flush=True)
    with open(dosya, "a", encoding="utf-8") as f:
        f.write(satir + "\n")


def temiz_eski_modeller(koru=5):
    """
    'model_bolum*.pth' dosyalarını tarihe göre sıralar,
    en yeni 'koru' adedini ve best_ modelini korur, gerisini siler.
    """
    dosyalar = [f for f in glob.glob("model_bolum*.pth")
                if not f.startswith("best_")]
    dosyalar.sort(key=os.path.getmtime)          # eskiden → yeniye
    silinecek = dosyalar[:-koru] if len(dosyalar) > koru else []
    for f in silinecek:
        try:
            os.remove(f)
            log_yaz(f"  [TEMIZLIK] Silindi: {f}")
        except Exception:
            pass


def run_gece_egitim(model_path=None):
    from car_env import CarEnv
    from ddqn_agent import DDQNAjani, EPSILON_BASLANGIC

    log_yaz("=" * 55)
    log_yaz("  GECE EGITIMI BASLADI")
    log_yaz(f"  Hedef: {BOLUM_SAYISI} bolum  |  Kayit: her {KAYIT_ARASI} bolumde")
    log_yaz("=" * 55)

    # Ortam ve ajan
    env   = CarEnv(render_mode=None)          # Headless — görsel YOK
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ajan  = DDQNAjani(env.observation_space.shape[0], env.action_space.n, cihaz)

    log_yaz(f"  Cihaz : {cihaz}")

    # Model yükle veya sıfırdan başla
    if model_path and os.path.exists(model_path):
        ajan.yukle(model_path)
        # Gece eğitimi için keşif oranını biraz açıyoruz (daha iyi öğrensin)
        if ajan.epsilon < 0.15:
            ajan.epsilon = 0.15
            log_yaz(f"  [EPSILON] Gece eğitimi için epsilon 0.15'e çekildi")
        log_yaz(f"  [DEVAM] Baz model: {model_path}  |  Epsilon={ajan.epsilon:.3f}")
    else:
        ajan.epsilon = EPSILON_BASLANGIC
        log_yaz("  [YENI] Sifirdan egitim baslıyor.")

    oduller, kayiplar = [], []
    en_iyi_skor = -float('inf')
    en_iyi_dosya = None
    baslangic = time.time()
    bolum = 1

    try:
        for bolum in range(1, BOLUM_SAYISI + 1):
            durum, _ = env.reset()
            toplam, kayip_list, bitti = 0.0, [], False

            adim = 0
            while not bitti:
                adim += 1
                eylem = ajan.eylem_sec(durum)
                yeni_durum, odul, term, trunc, _ = env.step(eylem)
                bitti = term or trunc or (adim >= MAX_ADIM_BOLUM)
                ajan.deneyim_ekle(durum, eylem, odul, yeni_durum, float(bitti))
                k = ajan.ogren()
                if k:
                    kayip_list.append(k)
                durum = yeni_durum
                toplam += odul

            oduller.append(toplam)
            ort_kayip = float(np.mean(kayip_list)) if kayip_list else 0.0
            kayiplar.append(ort_kayip)

            # ── Her 10 bölümde terminal çıktısı ─────────────
            if bolum % 10 == 0:
                son10 = np.mean(oduller[-10:])
                gecen = sure_format(time.time() - baslangic)
                kalan_est = (time.time() - baslangic) / bolum * (BOLUM_SAYISI - bolum)
                log_yaz(
                    f"  Bolum {bolum:>4}/{BOLUM_SAYISI} | "
                    f"Odul={toplam:>9.1f} | Ort10={son10:>9.1f} | "
                    f"eps={ajan.epsilon:.4f} | "
                    f"Gecen={gecen} | Kalan~{sure_format(kalan_est)}"
                )

            # ── Periyodik kayıt ──────────────────────────────
            if bolum % KAYIT_ARASI == 0:
                skor = int(np.mean(oduller[-10:])) if len(oduller) >= 10 else int(toplam)
                dosya_adi = f"model_bolum{bolum}_odul{skor}.pth"
                ajan.kaydet(dosya_adi)
                ajan.kaydet("ddqn_model.pth")   # Her zaman üstüne yazar

                # En iyi modeli ayrı sakla
                if EN_IYI_KAYIT and skor > en_iyi_skor:
                    en_iyi_skor = skor
                    en_iyi_dosya = f"best_model_odul{skor}.pth"
                    ajan.kaydet(en_iyi_dosya)
                    log_yaz(f"  ★ YENİ EN İYİ SKOR: {skor}  →  {en_iyi_dosya}")

                # Eski checkpoint'leri temizle (disk dolmasın)
                temiz_eski_modeller(koru=5)

        log_yaz("\n  [BITTI] Gece egitimi tamamlandi!")
        ajan.kaydet("ddqn_model_FINAL.pth")

    except KeyboardInterrupt:
        log_yaz(f"\n  [DURDURULDU] Bolum={bolum} | Kaydediliyor...")
        skor = int(np.mean(oduller[-10:])) if len(oduller) >= 10 else 0
        ajan.kaydet(f"model_durduruldu_bolum{bolum}_odul{skor}.pth")
        ajan.kaydet("ddqn_model.pth")

    finally:
        env.close()
        gecen_toplam = sure_format(time.time() - baslangic)
        log_yaz("=" * 55)
        log_yaz(f"  Toplam egitim suresi : {gecen_toplam}")
        log_yaz(f"  Tamamlanan bolum     : {bolum}/{BOLUM_SAYISI}")
        log_yaz(f"  En iyi skor          : {en_iyi_skor:.0f}")
        if en_iyi_dosya:
            log_yaz(f"  En iyi model         : {en_iyi_dosya}")
        log_yaz(f"  Log dosyasi          : {LOG_DOSYASI}")
        log_yaz("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gece Headless DDQN Egitimi")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Devam edilecek model dosyası (örn: model_bolum50_odul16584.pth)"
    )
    args = parser.parse_args()
    run_gece_egitim(model_path=args.model)
