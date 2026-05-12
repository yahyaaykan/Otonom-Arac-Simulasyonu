import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import os
import sys
import time

# 1. EN İYİ ÖĞRENME AYARLARI (Full Otomatik)
env_config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
    },
    "lanes_count": 4,           
    "vehicles_count": 25,       
    "duration": 60,             
    
    # --- KARARLILIK VE SÜZÜLME AYARLARI ---
    "simulation_frequency": 15,
    "policy_frequency": 2,         # Saniyede 2 karar: Sarsıntıyı bitirir, akıcılığı sağlar.
    
    # --- ÖDÜL HİYERARŞİSİ (Sıfır Hata Hedefi) ---
    "collision_reward": -500,      # KAZA: Çok ağır ceza (Kaçmayı öğrenmesi için).
    "on_road_reward": 8.0,         # ŞERİTTE KALMA: Mıknatıs etkisi (Dümdüz gider).
    "lane_change_reward": -2.0,    # ŞERİT DEĞİŞTİRME: Gereksizse yapmaz, gerekirse yapar.
    "high_speed_reward": 1.5,      # HIZ: Önü boşsa gaza basar.
    "reward_speed_range": [25, 35] 
}

env = gym.make('highway-v0', render_mode='human')
env.unwrapped.configure(env_config)

# 2. MODEL VE KAYIT SİSTEMİ
# ÖNEMLİ: Eğer araç hala saçmalıyorsa lütfen bu .zip dosyasını SİLMENİZİ öneririm.
model_path = "otomatik_usta_sofor.zip"
log_file = "surus_gunlugu.txt"

if os.path.exists(model_path):
    print("\n[SİSTEM] Kayıtlı beyin yüklendi. Öğrenme kaldığı yerden devam ediyor...")
    model = DQN.load(model_path, env=env)
else:
    print("\n[SİSTEM] SIFIRDAN EĞİTİM BAŞLIYOR. İlk turlar kaza yapabilir, izleyin...")
    # Öğrenme hızını (1e-4) çok hassas tuttuk ki "kalıcı" alışkanlıklar edinsin.
    model = DQN("MlpPolicy", env, verbose=0, learning_rate=1e-4, buffer_size=100000)

if not os.path.exists(log_file):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Tur No | Toplam Puan | Adım Sayısı | Sonuç\n" + "-"*55 + "\n")

# 3. KENDİ KENDİNE ÖĞRENME DÖNGÜSÜ
try:
    episode_count = 1
    while True:
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0
        step_num = 0
        
        print(f"\n=== TUR {episode_count} BAŞLADI (Yapay Zeka Kendini Geliştiriyor) ===")
        
        while not (done or truncated):
            # Kararlı sürüş için deterministic=True
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_num += 1
            
            # Dashboard
            action_names = ["SOL", "DÜZ", "SAĞ", "GAZ", "FREN"]
            current_action = action_names[int(action)]
            sys.stdout.write(f"\r[Adım: {step_num:3d}] Karar: {current_action:4s} | Puan: {reward:6.2f} | Toplam: {episode_reward:7.2f}")
            sys.stdout.flush()
            
            env.render()

        # TUR SONU ANALİZİ
        sonuc = "KAZA" if reward <= -50 else "BAŞARILI"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Tur {episode_count:3d} | {episode_reward:12.2f} | {step_num:10d} | {sonuc}\n")
        
        # DERİN ÖĞRENME (En önemli kısım)
        # Her tur sonunda 15 bin adım kafa yorarak "Neden kaza yaptım?" diye analiz eder.
        print(f"\n\n>>> {sonuc}! Hatalar analiz ediliyor ve beyne kazınıyor (Lütfen bekleyin)...")
        model.learn(total_timesteps=15000, reset_num_timesteps=False)
        model.save(model_path)
        
        episode_count += 1
        print("=" * 60)

except KeyboardInterrupt:
    print("\nSimülasyon durduruldu. Beyin 'otomatik_usta_sofor.zip' dosyasına kaydedildi.")
    env.close()