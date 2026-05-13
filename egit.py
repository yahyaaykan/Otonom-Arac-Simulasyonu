# -*- coding: utf-8 -*-
"""
Eğitim launcher — tek tıkla çalıştır.
Gorsel: render=True -> Pygame penceresi açılır.
         render=False -> sadece terminal çıktısı (hızlı).
"""
import sys
import os
import glob
import torch
import numpy as np
import pygame
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from car_env import CarEnv
from ddqn_agent import (
    DDQNAjani, EgitimYoneticisi,
    TOPLAM_BOLUM, GAMMA, OGRENME_HIZI,
    EPSILON_BASLANGIC, EPSILON_MIN, EPSILON_AZALMA,
    BUFFER_BOYUTU, BATCH_BOYUTU, HEDEF_AG_GUNCELLEME
)

# ─── AYARLAR ────────────────────────────────────────────────
# render=True  → Pygame penceresi ile görsel sürüş izle
# render=False → Görsel yok, ~5-10x daha hızlı eğitim
RENDER_MOD    = False   # ⚡ KAPALI = 5-10x hızlı eğitim! Demo için main.py kullan.
BOLUM_SAYISI  = 600
KAYIT_ARASI   = 50

# ESKI MODEL YUKLENMESIN: Yanlis politikayla egitilmis model varsa sil
# True = sifirdan egitim  |  False = varsa devam et
SIFIRDAN_BASLA = True

def bilgi_satiri():
    print("=" * 60, flush=True)
    print("  DUELING DOUBLE DQN — OTONOM ARAC EGITIMI", flush=True)
    print("=" * 60, flush=True)
    print(f"  Bolum Sayisi       : {BOLUM_SAYISI}", flush=True)
    print(f"  Gamma              : {GAMMA}", flush=True)
    print(f"  Ogrenme Hizi       : {OGRENME_HIZI}", flush=True)
    print(f"  Epsilon Baslangic  : {EPSILON_BASLANGIC}", flush=True)
    print(f"  Epsilon Min        : {EPSILON_MIN}", flush=True)
    print(f"  Epsilon Azalma     : {EPSILON_AZALMA}", flush=True)
    print(f"  Buffer Boyutu      : {BUFFER_BOYUTU}", flush=True)
    print(f"  Batch Boyutu       : {BATCH_BOYUTU}", flush=True)
    print(f"  Hedef Ag Guncelleme: {HEDEF_AG_GUNCELLEME} adim", flush=True)
    print(f"  Render Modu        : {'ACIK (Pygame)' if RENDER_MOD else 'KAPALI (Hizli)'}", flush=True)
    print(f"  CUDA               : {'VAR' if torch.cuda.is_available() else 'YOK (CPU)'}", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)

def _model_bilgi(dosya):
    """Model dosyasından boyut ve tarih bilgisi döndürür."""
    import time as _time
    try:
        boyut_mb = os.path.getsize(dosya) / (1024 * 1024)
        tarih = _time.strftime("%d.%m.%Y %H:%M", _time.localtime(os.path.getmtime(dosya)))
    except Exception:
        boyut_mb, tarih = 0.0, "?"
    return boyut_mb, tarih


def secim_menusu():
    pygame.init()
    screen = pygame.display.set_mode((660, 520))
    pygame.display.set_caption("Otonom Arac AI - Model Secici")

    font_baslik = pygame.font.SysFont("Impact",      34)
    font_buton  = pygame.font.SysFont("Arial Black", 16)
    font_kart   = pygame.font.SysFont("Consolas",    13, bold=True)
    font_bilgi  = pygame.font.SysFont("Consolas",    11)

    # .pth VE .zip dosyaların hepsini tara — en yeni önce
    modeller = sorted(
        glob.glob("*.pth") + glob.glob("*.zip"),
        key=os.path.getmtime, reverse=True
    )
    model_var     = len(modeller) > 0
    secili_index  = 0
    secim         = None
    secilen_model = None

    btn_sifir = pygame.Rect(40,  448, 270, 48)
    btn_devam = pygame.Rect(350, 448, 270, 48)
    btn_prev  = pygame.Rect(14,  190, 38,  190)
    btn_next  = pygame.Rect(608, 190, 38,  190)

    while secim is None:
        screen.fill((10, 14, 24))

        # Başlık
        baslik = font_baslik.render("OTONOM ARAC — MODEL SECiCi", True, (0, 220, 255))
        screen.blit(baslik, baslik.get_rect(center=(330, 36)))
        pygame.draw.line(screen, (0, 160, 210), (40, 60), (620, 60), 2)

        # Model kartı alanı
        kart = pygame.Rect(58, 78, 544, 340)
        pygame.draw.rect(screen, (18, 26, 42), kart, border_radius=14)
        pygame.draw.rect(screen, (0, 130, 190), kart, 2, border_radius=14)

        if model_var:
            model     = modeller[secili_index]
            boyut_mb, tarih = _model_bilgi(model)
            uzanti    = "PTH — DDQN (Aktif Format)" if model.endswith(".pth") else "ZIP — Eski / PPO"
            renk_uzanti = (80, 255, 120) if model.endswith(".pth") else (255, 160, 60)

            # Model adı
            ad = font_buton.render(model, True, (255, 220, 0))
            screen.blit(ad, ad.get_rect(center=(330, 112)))

            # Bilgi satırları
            satirlar = [
                ("FORMAT",    uzanti,                          renk_uzanti),
                ("BOYUT",     f"{boyut_mb:.2f} MB",            (200, 210, 230)),
                ("TARIH",     tarih,                           (200, 210, 230)),
                ("SIRALAMA",  f"{secili_index+1} / {len(modeller)} model",   (180, 200, 255)),
            ]
            for i, (etiket, deger, renk) in enumerate(satirlar):
                y = 150 + i * 52
                pygame.draw.rect(screen, (28, 38, 58), pygame.Rect(78, y - 6, 504, 38), border_radius=7)
                e_surf = font_kart.render(f"{etiket:<11}", True, (100, 160, 255))
                d_surf = font_kart.render(deger,           True, renk)
                screen.blit(e_surf, (94,  y + 4))
                screen.blit(d_surf, (270, y + 4))

            # Klavye ipucu
            ipucu = font_bilgi.render("< > ok tuslar veya butonla model degistir  |  ENTER: Devam et", True, (70, 100, 140))
            screen.blit(ipucu, ipucu.get_rect(center=(330, 392)))

            # Ok butonları
            pygame.draw.rect(screen, (35, 55, 90), btn_prev, border_radius=8)
            pygame.draw.rect(screen, (35, 55, 90), btn_next, border_radius=8)
            screen.blit(font_buton.render("<", True, (180, 220, 255)),
                        font_buton.render("<", True, (0,0,0)).get_rect(center=btn_prev.center))
            screen.blit(font_buton.render(">", True, (180, 220, 255)),
                        font_buton.render(">", True, (0,0,0)).get_rect(center=btn_next.center))
        else:
            uyari = font_buton.render("Kayitli model bulunamadi!", True, (255, 80, 80))
            screen.blit(uyari, uyari.get_rect(center=(330, 210)))
            acik = font_bilgi.render("Egitim bittikten sonra .pth dosyalari burada gorunecek.", True, (120, 120, 140))
            screen.blit(acik, acik.get_rect(center=(330, 240)))

        # Alt butonlar
        pygame.draw.rect(screen, (170, 35, 35), btn_sifir, border_radius=10)
        t1 = font_buton.render("SIFIRDAN BASLA", True, (255, 255, 255))
        screen.blit(t1, t1.get_rect(center=btn_sifir.center))

        renk_devam = (30, 155, 55) if model_var else (55, 65, 78)
        pygame.draw.rect(screen, renk_devam, btn_devam, border_radius=10)
        t2 = font_buton.render("BU MODELLE DEVAM ET" if model_var else "MODEL YOK", True, (255, 255, 255))
        screen.blit(t2, t2.get_rect(center=btn_devam.center))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT  and model_var:
                    secili_index = (secili_index - 1) % len(modeller)
                elif event.key == pygame.K_RIGHT and model_var:
                    secili_index = (secili_index + 1) % len(modeller)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and model_var:
                    secim = "DEVAM"; secilen_model = modeller[secili_index]
                elif event.key == pygame.K_ESCAPE:
                    secim = "SIFIR"
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_sifir.collidepoint(event.pos):
                    secim = "SIFIR"
                elif btn_devam.collidepoint(event.pos) and model_var:
                    secim = "DEVAM"; secilen_model = modeller[secili_index]
                elif model_var and btn_prev.collidepoint(event.pos):
                    secili_index = (secili_index - 1) % len(modeller)
                elif model_var and btn_next.collidepoint(event.pos):
                    secili_index = (secili_index + 1) % len(modeller)

    pygame.quit()
    return secim == "SIFIR", secilen_model

def main():
    bilgi_satiri()

    # Menüden seçimi al (Env oluşturulmadan ÖNCE yapılmalı ki Pygame çökmesin)
    secilen_model = None
    if RENDER_MOD:
        sifirdan_secildi, secilen_model = secim_menusu()
    else:
        sifirdan_secildi = SIFIRDAN_BASLA

    env = CarEnv(render_mode="human" if RENDER_MOD else None)
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    durum_boyutu = env.observation_space.shape[0]
    eylem_sayisi = env.action_space.n

    ajan = DDQNAjani(durum_boyutu, eylem_sayisi, cihaz)

    # Sifirdan mi, yoksa kayitli modelden mi?
    if sifirdan_secildi:
        if os.path.exists("ddqn_model.pth"):
            os.remove("ddqn_model.pth")
        print("  [YENI] Sifirdan egitim basliyor.", flush=True)
        ajan.epsilon = EPSILON_BASLANGIC  # Sabit 1.0 ile tam keşif modu
        print(f"  Epsilon sifirdan: {ajan.epsilon}", flush=True)
    else:
        if secilen_model and os.path.exists(secilen_model):
            ajan.yukle(secilen_model)
        else:
            ajan.yukle() # Varsayilan
        # Modeli yukledikten sonra epsilon cok dusukse artir
        if ajan.epsilon < 0.30:
            ajan.epsilon = 0.30
            print(f"  [EPSILON RESET] Kesif icin epsilon 0.30 yapildi.", flush=True)

    bolum_odulleri  = []
    bolum_kayiplari = []

    try:
        for bolum in range(1, BOLUM_SAYISI + 1):
            durum, _ = env.reset()
            
            # HUD'a güncel bilgileri doğrudan aktar
            env.info_episode = bolum
            env.info_epsilon = ajan.epsilon
                
            toplam_odul = 0.0
            bolum_kayip = []
            bitti = False

            while not bitti:
                eylem = ajan.eylem_sec(durum)
                yeni_durum, odul, terminated, truncated, _ = env.step(eylem)
                bitti = terminated or truncated

                ajan.deneyim_ekle(durum, eylem, odul, yeni_durum, float(bitti))
                kayip = ajan.ogren()
                if kayip is not None:
                    bolum_kayip.append(kayip)
                    env.info_loss = kayip

                durum = yeni_durum
                toplam_odul += odul

            bolum_odulleri.append(toplam_odul)
            ort_kayip = float(np.mean(bolum_kayip)) if bolum_kayip else 0.0
            bolum_kayiplari.append(ort_kayip)

            # Her bölümü yaz (flush ile aninda goster)
            if bolum % 10 == 0:
                son10 = np.mean(bolum_odulleri[-10:])
                print(
                    f"  Bolum {bolum:>4}/{BOLUM_SAYISI} | "
                    f"Odul={toplam_odul:>9.1f} | "
                    f"Ort10={son10:>9.1f} | "
                    f"eps={ajan.epsilon:.4f} | "
                    f"Kayip={ort_kayip:.5f}",
                    flush=True
                )

            # Periyodik kayit + grafik
            if bolum % KAYIT_ARASI == 0:
                son10_skor = int(np.mean(bolum_odulleri[-10:])) if len(bolum_odulleri) >= 10 else int(toplam_odul)
                ozel_isim = f"model_bolum{bolum}_odul{son10_skor}.pth"
                ajan.kaydet(ozel_isim)
                # Düz ismi de güncel tut
                ajan.kaydet("ddqn_model.pth")
                _grafik_ciz(bolum_odulleri, bolum_kayiplari, ara=True)

        # Egitim sonu normal kayit
        ajan.kaydet("ddqn_model.pth")
        _grafik_ciz(bolum_odulleri, bolum_kayiplari, ara=False)
        print("\n  [BITTI] Egitim tamamlandi!", flush=True)

    except KeyboardInterrupt:
        # CTRL+C ILE DURDURULURSA OTOMATIK KAYIT YAP
        print("\n\n  [UYARI] Terminal durduruldu (CTRL+C)!", flush=True)
        print("  [KAYIT] O ana kadarki model bilgileri guvenli sekilde kaydediliyor...", flush=True)
        # Hangi bölümde durdurduğunu isme ekle ki "gir-çık" yaptığın anlaşılsın
        isim = f"model_manuel_bolum{bolum}.pth"
        ajan.kaydet(isim)
        _grafik_ciz(bolum_odulleri, bolum_kayiplari, ara=True)
        print(f"  [BASARILI] Kayit tamamlandi: {isim}\n", flush=True)

    finally:
        env.close()


def _grafik_ciz(bolum_odulleri, bolum_kayiplari, ara=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("DDQN Egitim Analizi - Otonom Arac", fontsize=14, fontweight="bold")

    ax1.plot(bolum_odulleri, alpha=0.4, color="steelblue", label="Ham Odul")
    if len(bolum_odulleri) >= 20:
        pencere = 20
        hort = np.convolve(bolum_odulleri, np.ones(pencere) / pencere, mode="valid")
        ax1.plot(range(pencere - 1, len(bolum_odulleri)), hort,
                 color="crimson", linewidth=2, label=f"{pencere}-Bolum Ort.")
    ax1.set_ylabel("Toplam Odul")
    ax1.set_xlabel("Bolum")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(bolum_kayiplari, alpha=0.5, color="darkorange", label="Bolum Kaybi")
    ax2.set_ylabel("Ortalama Kayip (Huber)")
    ax2.set_xlabel("Bolum")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    dosya = "egitim_analizi_ara.png" if ara else "egitim_analizi_FINAL.png"
    plt.savefig(dosya, dpi=150)
    plt.close()
    print(f"  [GRAFIK] {dosya} kaydedildi.", flush=True)


if __name__ == "__main__":
    main()
