# -*- coding: utf-8 -*-
"""
Ana Giriş Noktası — DDQN Otonom Araç Simülasyonu
Eğitim için: egit.py
İzleme için: bu dosya
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
from car_env import CarEnv
from ddqn_agent import DDQNAjani, KAYIT_DOSYASI


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    print("""
    ====================================================
      OTONOM ARAC YAPAY ZEKA SIMULASYONU (V14.0)
      Algoritma : Dueling Double DQN
      Ortam     : 5 Seritli Otoyol (CarEnv)
    ====================================================
    """)


def izle(bolum_sayisi: int = 10):
    """Eğitilmiş DDQN modelini görsel olarak çalıştırır."""
    model_dosyasi = KAYIT_DOSYASI

    # .pth dosyası yoksa kullanıcıya sor
    if not os.path.exists(model_dosyasi):
        pth_dosyalari = [f for f in os.listdir('.') if f.endswith('.pth')]
        if pth_dosyalari:
            print("\n  Mevcut modeller:")
            for i, f in enumerate(pth_dosyalari):
                print(f"    [{i}] {f}")
            try:
                secim = int(input("  Model numarasi secin: "))
                model_dosyasi = pth_dosyalari[secim]
            except (ValueError, IndexError):
                print("  [HATA] Gecersiz secim.")
                return
        else:
            print("\n  [HATA] Hic kayitli model bulunamadi. Once 'egit.py' ile egitim yapin.")
            time.sleep(2)
            return

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CarEnv(render_mode="human")
    durum_boyutu = env.observation_space.shape[0]
    eylem_sayisi = env.action_space.n

    ajan = DDQNAjani(durum_boyutu, eylem_sayisi, cihaz)
    ajan.yukle(model_dosyasi)
    ajan.epsilon = 0.0  # Tam deterministik — keşif kapalı

    print(f"\n  Model: {model_dosyasi}  |  Cihaz: {cihaz}")
    print(f"  {bolum_sayisi} bolum izlenecek...\n")

    try:
        for bolum in range(1, bolum_sayisi + 1):
            durum, _ = env.reset()
            toplam_odul = 0.0
            bitti = False
            adim = 0

            while not bitti:
                eylem = ajan.eylem_sec(durum)
                durum, odul, terminated, truncated, _ = env.step(eylem)
                toplam_odul += odul
                bitti = terminated or truncated
                adim += 1

            sonuc = "KAZA" if terminated else "TAMAMLANDI"
            print(f"  Bolum {bolum:>3}: {sonuc:>12} | Toplam Odul: {toplam_odul:>9.1f} | Adim: {adim}")
    except KeyboardInterrupt:
        print("\n  [DURDURULDU] Kullanici tarafindan kesildi.")
    finally:
        env.close()


def egitim_baslat():
    """Kullanıcıyı egit.py çalıştırmaya yönlendirir."""
    print("""
  Egitim icin:
    python egit.py

  egit.py, Pygame onizleme menusu ile birlikte gelir:
    - Sifirdan baslat
    - Kayitli modelden devam et
    - Model secimi (oklar ile)
    """)
    input("  Devam etmek icin Enter'a basin...")


def main():
    while True:
        clear_screen()
        print_banner()

        cuda_durum = "VAR ✓" if torch.cuda.is_available() else "YOK (CPU)"
        model_var = os.path.exists(KAYIT_DOSYASI)
        model_durum = f"Bulundu: {KAYIT_DOSYASI}" if model_var else "Kayitli model yok"

        print(f"  CUDA   : {cuda_durum}")
        print(f"  Model  : {model_durum}")
        print()
        print("  1) Modeli Izle (Gorsel Test)")
        print("  2) Egitim Baslatma Rehberi")
        print("  3) Cikis")
        print()

        try:
            secim = input("  Seciminiz: ").strip()
            if secim == "1":
                izle(bolum_sayisi=10)
            elif secim == "2":
                egitim_baslat()
            elif secim == "3":
                break
        except EOFError:
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n  [HATA] {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
