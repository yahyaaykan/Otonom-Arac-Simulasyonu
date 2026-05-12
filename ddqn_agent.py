# -*- coding: utf-8 -*-
"""
=============================================================
  OTONOM ARAC - DOUBLE DQN (DDQN) + DUELING MIMARISI
  Proje: CarEnv Otoyol Simulasyonu
  Algoritma: Double DQN + Dueling Network + Experience Replay
=============================================================
"""
import sys
import io
# Windows terminallerde Turkce karakter sorunu onleme
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import sys as _sys
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# HİPERPARAMETRELER
# ─────────────────────────────────────────────────────────────
# gamma (İskonto Faktörü): 0.99 → Ajan uzun vadeli ödülleri
#   önemser; yakın + uzak kazanımları dengeler.
GAMMA = 0.99

# epsilon (Keşif Oranı): Başta %100 rastgele, eğitim ilerledikçe
#   azalır. Ajan yavaş yavaş öğrendiğini kullanmaya başlar.
EPSILON_BASLANGIC = 1.0
EPSILON_MIN       = 0.05
EPSILON_AZALMA    = 0.9995   # Her adımda çarpılan katsayı

# learning_rate: 5e-4 → Adam optimizer için dengeli öğrenme adımı.
#   Çok büyük → ıraksama; çok küçük → yavaş öğrenme.
OGRENME_HIZI = 5e-4

# Replay Buffer: 50.000 geçmiş deneyim saklanır. Büyük buffer
#   çeşitlilik sağlar, bağımlılığı kırar.
BUFFER_BOYUTU = 50_000

# Batch: Her güncellemede 128 örnek çekilir. Küçük batch →
#   gürültülü; büyük batch → yavaş ama kararlı.
BATCH_BOYUTU = 128

# Hedef ağ her 500 adımda bir güncellenir (hard update).
#   Sık güncelleme → kararsız eğitim.
HEDEF_AG_GUNCELLEME = 500

# Karar frekansı: CarEnv'in step() çağrısı = 1 karar.
#   Gerçek zamanlı stabilite env tarafından yönetilir.
TOPLAM_BOLUM = 800
KAYIT_DOSYASI = "ddqn_model.pth"

# ─────────────────────────────────────────────────────────────
# KATMAN: DUELING DQN SINIRI AĞACI
# ─────────────────────────────────────────────────────────────
class DuelingDQNAgi(nn.Module):
    """
    Dueling Mimari:
      Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

    V(s)    → Durum değeri (eylemden bağımsız)
    A(s,a)  → Avantaj (bu eylemin diğerlerine göre üstünlüğü)

    Neden Dueling?
      Standart DQN her eylem için ayrı Q değeri öğrenir.
      Dueling, durumun genel değerini ayrıca öğrenerek
      daha kararlı ve hızlı yakınsama sağlar.
    """
    def __init__(self, giris_boyutu: int, cikis_boyutu: int):
        super(DuelingDQNAgi, self).__init__()

        # Paylaşılan özellik çıkarıcı katmanlar
        self.ozellik_katmanlari = nn.Sequential(
            nn.Linear(giris_boyutu, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Değer akışı: V(s) → tek sayı
        self.deger_akisi = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Avantaj akışı: A(s,a) → eylem sayısı kadar
        self.avantaj_akisi = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, cikis_boyutu),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ozellikler = self.ozellik_katmanlari(x)
        deger      = self.deger_akisi(ozellikler)        # (batch, 1)
        avantaj    = self.avantaj_akisi(ozellikler)      # (batch, n_eylem)

        # Dueling birleşim formülü
        q_degerleri = deger + (avantaj - avantaj.mean(dim=1, keepdim=True))
        return q_degerleri


# ─────────────────────────────────────────────────────────────
# KATMAN: DENEYİM TEKRAR TAMPONU (EXPERIENCE REPLAY)
# ─────────────────────────────────────────────────────────────
class DeneyimTamponu:
    """
    Experience Replay Buffer:
      Ajan her adımda (durum, eylem, ödül, yeni_durum, bitti)
      demetini tampona yazar. Eğitimde rastgele mini-batch
      örneklenerek örnekler arası korelasyon kırılır.
      Bu, eğitimi kararlı kılan temel mekanizmadır.
    """
    def __init__(self, kapasite: int):
        self.tampon = deque(maxlen=kapasite)

    def ekle(self, durum, eylem, odul, yeni_durum, bitti):
        self.tampon.append((durum, eylem, odul, yeni_durum, bitti))

    def ornekle(self, boyut: int):
        return random.sample(self.tampon, boyut)

    def __len__(self):
        return len(self.tampon)


# ─────────────────────────────────────────────────────────────
# KATMAN: DDQN AJANI
# ─────────────────────────────────────────────────────────────
class DDQNAjani:
    """
    Double DQN Ajani:
      Standart DQN'de hedef hesaplanırken aynı ağ hem eylem
      seçer hem değerlendirir → aşırı tahmin hatası (overestimation).

      Double DQN çözümü:
        1. Çevrimiçi ağ   → en iyi eylemi SEÇER
        2. Hedef ağ       → seçilen eylemin Q değerini HESAPLAR
      Bu ayrım overestimation bias'ını belirgin ölçüde azaltır.
    """

    def __init__(self, durum_boyutu: int, eylem_sayisi: int, cihaz: torch.device):
        self.durum_boyutu = durum_boyutu
        self.eylem_sayisi = eylem_sayisi
        self.cihaz        = cihaz
        self.epsilon      = EPSILON_BASLANGIC
        self.adim_sayaci  = 0

        # Çevrimiçi ağ: her adımda güncellenir
        self.cevrimici_ag = DuelingDQNAgi(durum_boyutu, eylem_sayisi).to(cihaz)

        # Hedef ağ: belirli aralıklarla kopyalanır, eğitimi stabilize eder
        self.hedef_ag = DuelingDQNAgi(durum_boyutu, eylem_sayisi).to(cihaz)
        self.hedef_ag.load_state_dict(self.cevrimici_ag.state_dict())
        self.hedef_ag.eval()

        self.optimizer  = optim.Adam(self.cevrimici_ag.parameters(), lr=OGRENME_HIZI)
        self.kayip_fnk  = nn.SmoothL1Loss()   # Huber loss: aykırı değerlere dayanıklı
        self.tampon     = DeneyimTamponu(BUFFER_BOYUTU)

        # Loglama
        self.kayip_gecmisi = []
        self.epsilon_gecmisi = []

    def eylem_sec(self, durum: np.ndarray) -> int:
        """
        Epsilon-Greedy Politikası:
          - epsilon olasılıkla rastgele eylem (keşif)
          - (1-epsilon) olasılıkla en yüksek Q değerli eylem (sömürü)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.eylem_sayisi - 1)

        durum_tensor = torch.FloatTensor(durum).unsqueeze(0).to(self.cihaz)
        with torch.no_grad():
            q_degerleri = self.cevrimici_ag(durum_tensor)
        return int(q_degerleri.argmax(dim=1).item())

    def deneyim_ekle(self, durum, eylem, odul, yeni_durum, bitti):
        self.tampon.ekle(durum, eylem, odul, yeni_durum, bitti)

    def ogren(self) -> float | None:
        """
        Double DQN Güncelleme Adımı:
          Bellman hedefi: r + γ * Q_hedef(s', argmax_a Q_online(s', a))
          Kayıp: Huber(Q_online(s,a) - Bellman_hedefi)
        """
        if len(self.tampon) < BATCH_BOYUTU:
            return None

        self.adim_sayaci += 1

        # Mini-batch örnekle
        ornekler    = self.tampon.ornekle(BATCH_BOYUTU)
        durumlar    = torch.FloatTensor(np.array([o[0] for o in ornekler])).to(self.cihaz)
        eylemler    = torch.LongTensor([o[1] for o in ornekler]).unsqueeze(1).to(self.cihaz)
        oduller     = torch.FloatTensor([o[2] for o in ornekler]).unsqueeze(1).to(self.cihaz)
        yeni_durumlar = torch.FloatTensor(np.array([o[3] for o in ornekler])).to(self.cihaz)
        bitti_flags = torch.FloatTensor([o[4] for o in ornekler]).unsqueeze(1).to(self.cihaz)

        # Mevcut Q değerleri (seçilen eylemler için)
        mevcut_q = self.cevrimici_ag(durumlar).gather(1, eylemler)

        # Double DQN hedef hesabı
        with torch.no_grad():
            # Adım 1: Çevrimiçi ağ → en iyi eylemi seç
            en_iyi_eylemler = self.cevrimici_ag(yeni_durumlar).argmax(dim=1, keepdim=True)
            # Adım 2: Hedef ağ → seçilen eylemin değerini hesapla
            hedef_q = self.hedef_ag(yeni_durumlar).gather(1, en_iyi_eylemler)
            # Bellman denklemi
            bellman_hedef = oduller + GAMMA * hedef_q * (1.0 - bitti_flags)

        # Kayıp ve geri yayılım
        kayip = self.kayip_fnk(mevcut_q, bellman_hedef)
        self.optimizer.zero_grad()
        kayip.backward()
        # Gradient clipping: patlamayı önler
        torch.nn.utils.clip_grad_norm_(self.cevrimici_ag.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Epsilon azalt
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_AZALMA

        # Hedef ağı periyodik güncelle (hard copy)
        if self.adim_sayaci % HEDEF_AG_GUNCELLEME == 0:
            self.hedef_ag.load_state_dict(self.cevrimici_ag.state_dict())

        kayip_deger = kayip.item()
        self.kayip_gecmisi.append(kayip_deger)
        return kayip_deger

    def kaydet(self, dosya: str = KAYIT_DOSYASI):
        torch.save({
            "cevrimici_ag": self.cevrimici_ag.state_dict(),
            "hedef_ag":     self.hedef_ag.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "epsilon":      self.epsilon,
            "adim":         self.adim_sayaci,
        }, dosya)
        print(f"  [KAYIT] Model kaydedildi → {dosya}")

    def yukle(self, dosya: str = KAYIT_DOSYASI):
        if not os.path.exists(dosya):
            print(f"  [UYARI] '{dosya}' bulunamadı, sıfırdan başlanıyor.")
            return
        kayit = torch.load(dosya, map_location=self.cihaz)
        self.cevrimici_ag.load_state_dict(kayit["cevrimici_ag"])
        self.hedef_ag.load_state_dict(kayit["hedef_ag"])
        self.optimizer.load_state_dict(kayit["optimizer"])
        self.epsilon    = kayit.get("epsilon", EPSILON_MIN)
        self.adim_sayaci = kayit.get("adim", 0)
        print(f"  [YÜKLEME] Model yüklendi. Epsilon={self.epsilon:.3f}")


# ─────────────────────────────────────────────────────────────
# KATMAN: EĞİTİM YÖNETİCİSİ
# ─────────────────────────────────────────────────────────────
class EgitimYoneticisi:
    """
    Eğitim döngüsünü, loglama ve grafik çizimini yönetir.
    """

    def __init__(self, env_sinifi, render: bool = False):
        self.env       = env_sinifi(render_mode="human" if render else None)
        self.render    = render
        self.cihaz     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        durum_boyutu  = self.env.observation_space.shape[0]
        eylem_sayisi  = self.env.action_space.n

        self.ajan = DDQNAjani(durum_boyutu, eylem_sayisi, self.cihaz)
        print(f"\n  Cihaz  : {self.cihaz}")
        print(f"  Durum  : {durum_boyutu}  |  Eylem : {eylem_sayisi}")
        print(f"  Mimari : Dueling Double DQN\n")

        # Log listeleri
        self.bolum_odulleri  = []
        self.bolum_kayiplari = []

    def egit(self, bolum_sayisi: int = TOPLAM_BOLUM, kayit_arasi: int = 50):
        print("=" * 55)
        print("  EGITIM BASLADI")
        print("=" * 55)

        for bolum in range(1, bolum_sayisi + 1):
            durum, _ = self.env.reset()
            toplam_odul  = 0.0
            bolum_kayip  = []
            bitti = False

            while not bitti:
                eylem  = self.ajan.eylem_sec(durum)
                yeni_durum, odul, terminated, truncated, _ = self.env.step(eylem)
                bitti  = terminated or truncated

                self.ajan.deneyim_ekle(durum, eylem, odul, yeni_durum, float(bitti))
                kayip = self.ajan.ogren()
                if kayip is not None:
                    bolum_kayip.append(kayip)

                durum       = yeni_durum
                toplam_odul += odul

            self.bolum_odulleri.append(toplam_odul)
            ort_kayip = float(np.mean(bolum_kayip)) if bolum_kayip else 0.0
            self.bolum_kayiplari.append(ort_kayip)
            self.ajan.epsilon_gecmisi.append(self.ajan.epsilon)

            # Terminal ciktisi
            if bolum % 10 == 0:
                son10 = np.mean(self.bolum_odulleri[-10:])
                print(f"  Bolum {bolum:>4}/{bolum_sayisi} | "
                      f"Odul={toplam_odul:>8.1f} | "
                      f"Ort10={son10:>8.1f} | "
                      f"eps={self.ajan.epsilon:.3f} | "
                      f"Kayip={ort_kayip:.4f}")

            # Periyodik kayıt
            if bolum % kayit_arasi == 0:
                self.ajan.kaydet()
                self._grafik_ciz(ara=True)

        # Eğitim sonu
        self.ajan.kaydet()
        self._grafik_ciz(ara=False)
        self.env.close()
        print("\n  [BITTI] Egitim tamamlandi.")

    def izle(self, bolum_sayisi: int = 5):
        """Eğitilmiş ajanı görsel olarak test eder."""
        self.ajan.yukle()
        self.ajan.epsilon = 0.0   # Keşif kapalı → tam deterministik

        env_izle = type(self.env)(render_mode="human")
        for b in range(1, bolum_sayisi + 1):
            durum, _ = env_izle.reset()
            toplam   = 0.0
            bitti    = False
            while not bitti:
                eylem = self.ajan.eylem_sec(durum)
                durum, odul, terminated, truncated, _ = env_izle.step(eylem)
                toplam += odul
                bitti   = terminated or truncated
            print(f"  İzleme Bölüm {b}: Toplam Ödül = {toplam:.1f}")
        env_izle.close()

    def _grafik_ciz(self, ara: bool = False):
        """Ödül ve kayıp grafiklerini çizer, PNG olarak kaydeder."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("DDQN Egitim Analizi - Otonom Arac", fontsize=14, fontweight="bold")

        # ── Ödül Grafiği ──────────────────────────────────────
        ax1.plot(self.bolum_odulleri, alpha=0.4, color="steelblue", label="Ham Odul")
        if len(self.bolum_odulleri) >= 20:
            pencere = 20
            hareketli_ort = np.convolve(
                self.bolum_odulleri, np.ones(pencere) / pencere, mode="valid"
            )
            ax1.plot(range(pencere - 1, len(self.bolum_odulleri)),
                     hareketli_ort, color="crimson", linewidth=2,
                     label=f"{pencere}-Bolum Hareketli Ortalama")
        ax1.set_ylabel("Toplam Odul")
        ax1.set_xlabel("Bölüm")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # ── Kayıp Grafiği ─────────────────────────────────────
        ax2.plot(self.bolum_kayiplari, alpha=0.5, color="darkorange", label="Bolum Kaybi")
        ax2.set_ylabel("Ortalama Kayip (Huber)")
        ax2.set_xlabel("Bölüm")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        cikti = "egitim_analizi_ara.png" if ara else "egitim_analizi_FINAL.png"
        plt.savefig(cikti, dpi=150)
        plt.close()
        print(f"  [GRAFIK] {cikti} kaydedildi.")


# ─────────────────────────────────────────────────────────────
# PROGRAM GİRİŞ NOKTASI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from car_env import CarEnv

    mod = input("Mod secin [E]git / [I]zle: ").strip().lower()

    if mod in ("e", "egit"):
        render_egitim = input("Egitim sirasinda gorsel acik olsun mu? [e/h]: ").strip().lower() == "e"
        yonetici = EgitimYoneticisi(CarEnv, render=render_egitim)
        yonetici.egit(bolum_sayisi=TOPLAM_BOLUM, kayit_arasi=50)

    elif mod in ("i", "izle"):
        yonetici = EgitimYoneticisi(CarEnv, render=True)
        yonetici.ajan.yukle()
        yonetici.izle(bolum_sayisi=10)

    else:
        print("Gecersiz secim. 'e' veya 'i' girin.")
