# -*- coding: utf-8 -*-
"""
Eğitim launcher — tek tıkla çalıştır.
Gorsel: render=True -> Pygame penceresi açılır.
         render=False -> sadece terminal çıktısı (hızlı).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from car_env import CarEnv
from ddqn_agent import (
    DDQNAjani, EgitimYoneticisi,
    TOPLAM_BOLUM, GAMMA, OGRENME_HIZI,
    EPSILON_BASLANGIC, EPSILON_MIN, EPSILON_AZALMA,
    BUFFER_BOYUTU, BATCH_BOYUTU, HEDEF_AG_GUNCELLEME
)
import torch, numpy as np
import pygame
import glob
import os

# ─── AYARLAR ────────────────────────────────────────────────
# render=True  → Pygame penceresi ile görsel sürüş izle
# render=False → Görsel yok, ~5-10x daha hızlı eğitim
RENDER_MOD    = True
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

def secim_menusu():
    pygame.init()
    screen = pygame.display.set_mode((600, 450))
    pygame.display.set_caption("AI Otonom Sürüş - Eğitim Launcher")
    font = pygame.font.SysFont("Arial Black", 20)
    font_small = pygame.font.SysFont("Consolas", 14)
    font_tiny = pygame.font.SysFont("Consolas", 12)
    
    # Buton tanımları (x, y, w, h)
    btn_sifir = pygame.Rect(100, 140, 400, 50)
    btn_devam = pygame.Rect(100, 210, 400, 50)
    btn_prev  = pygame.Rect(100, 280, 50, 40)
    btn_next  = pygame.Rect(450, 280, 50, 40)
    
    modeller = glob.glob("*.pth")
    # En yeni dosyalar en üstte olsun
    modeller.sort(key=os.path.getmtime, reverse=True)
    
    secili_index = 0
    model_var = len(modeller) > 0
    secim = None
    secilen_model = None
    
    while secim is None:
        screen.fill((15, 20, 30))
        
        title = font.render("Otonom Sürüş Eğitimi", True, (0, 255, 255))
        screen.blit(title, (170, 40))
        
        info_txt = font_small.render("Modeli egitmek icin bir secenek belirleyin:", True, (200, 200, 200))
        screen.blit(info_txt, (120, 90))
        
        # Sıfırdan Başla
        pygame.draw.rect(screen, (200, 50, 50), btn_sifir, border_radius=10)
        t_sifir = font.render("SIFIRDAN BASLA", True, (255, 255, 255))
        screen.blit(t_sifir, (210, 150))
        
        # Devam Et
        renk_devam = (50, 200, 50) if model_var else (100, 100, 100)
        pygame.draw.rect(screen, renk_devam, btn_devam, border_radius=10)
        yazi = "SECILI MODELLE DEVAM ET" if model_var else "Kayitli Model Yok!"
        t_devam = font.render(yazi, True, (255, 255, 255))
        screen.blit(t_devam, (160 if model_var else 180, 220))
        
        if model_var:
            pygame.draw.rect(screen, (80, 80, 100), btn_prev, border_radius=5)
            pygame.draw.rect(screen, (80, 80, 100), btn_next, border_radius=5)
            screen.blit(font.render("<", True, (255, 255, 255)), (115, 285))
            screen.blit(font.render(">", True, (255, 255, 255)), (465, 285))
            
            guncel_model = modeller[secili_index]
            model_txt = font_small.render(f"Model: {guncel_model}", True, (255, 255, 0))
            text_rect = model_txt.get_rect(center=(300, 300))
            screen.blit(model_txt, text_rect)
            
            detay = font_tiny.render(f"Toplam {len(modeller)} model bulundu. Oklarla degistirin.", True, (150, 150, 150))
            detay_rect = detay.get_rect(center=(300, 340))
            screen.blit(detay, detay_rect)
            
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if btn_sifir.collidepoint(event.pos):
                        secim = "SIFIR"
                    elif btn_devam.collidepoint(event.pos) and model_var:
                        secim = "DEVAM"
                        secilen_model = modeller[secili_index]
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
        ajan.epsilon = 0.95
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

    for bolum in range(1, BOLUM_SAYISI + 1):
        durum, _ = env.reset()
        
        # HUD'a güncel bilgileri aktar
        if hasattr(env, 'unwrapped'):
            env.unwrapped.info_episode = bolum
            env.unwrapped.info_epsilon = ajan.epsilon
            
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
                if hasattr(env, 'unwrapped'):
                    env.unwrapped.info_loss = kayip

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
            # Düz ismi de güncel tut (isteğe bağlı)
            ajan.kaydet("ddqn_model.pth")
            _grafik_ciz(bolum_odulleri, bolum_kayiplari, ara=True)

    # Egitim sonu
    ajan.kaydet()
    _grafik_ciz(bolum_odulleri, bolum_kayiplari, ara=False)
    env.close()
    print("\n  [BITTI] Egitim tamamlandi!", flush=True)


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
