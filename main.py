import os
import sys
import time
import warnings
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from car_env import CarEnv

# Gereksiz uyarıları gizle
warnings.filterwarnings("ignore")

# --- AYARLAR ---
MODEL_NAME = "otonom_arac_beyni_v14"
TIMESTEPS_PER_BATCH = 50000 

class PygameEventCallback(BaseCallback):
    """
    PPO ogrenirken Pygame penceresinin 'Yanit Vermiyor' demesini engeller.
    Her adimda Windows mesajlarini isler.
    """
    def __init__(self, verbose=0):
        super(PygameEventCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Pygame eventlerini her adimda pompala (Donmayi onler)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False # Egitimi durdur
        return True

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    print("""
    ====================================================
      OTONOM ARAC YAPAY ZEKA SIMULASYONU (V14.0)
    ====================================================
    [Gelisim: 5 Seritli Akilli Koridor Yönetimi]
    """)

def train():
    try:
        env = CarEnv(render_mode="human")
    except Exception as e:
        print(f"\n[HATA] Ortam baslatilamadi: {e}")
        input("Cikmak icin Enter...")
        return

    try:
        # Callback olustur
        callback = PygameEventCallback()
        
        if os.path.exists(f"{MODEL_NAME}.zip"):
            print(f"\n[!] Model yukleniyor...")
            model = PPO.load(MODEL_NAME, env=env)
        else:
            print("\n[+] Sifirdan egitim baslatiliyor...")
            model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003)
        
        print("\nEgitim basladi! Pencere artik donmayacaktir.")
        while True:
            model.learn(total_timesteps=TIMESTEPS_PER_BATCH, 
                        reset_num_timesteps=False,
                        callback=callback)
            model.save(MODEL_NAME)
            print(f"[OK] {TIMESTEPS_PER_BATCH} adim kaydedildi.")
            
    except KeyboardInterrupt:
        print("\n[!] Durduruldu.")
    except Exception as e:
        print(f"\n[HATA] Beklenmedik hata: {e}")
        input("Enter'a basin...")
    finally:
        env.close()

def watch():
    if not os.path.exists(f"{MODEL_NAME}.zip"):
        print("\n[!] Model bulunamadi.")
        time.sleep(2)
        return

    try:
        env = CarEnv(render_mode="human")
        model = PPO.load(MODEL_NAME, env=env)
        obs, info = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    except Exception as e:
        print(f"\n[HATA] {e}")
        input("Enter...")
    finally:
        env.close()

def main():
    while True:
        clear_screen()
        print_banner()
        print("1) AI Egitimini Baslat (Donma Korumali)")
        print("2) AI Izle")
        print("3) Cikis")
        
        try:
            secim = input("\nSeciminiz: ")
            if secim == "1": train()
            elif secim == "2": watch()
            elif secim == "3": break
        except EOFError:
            break
        except Exception:
            continue

if __name__ == "__main__":
    main()
