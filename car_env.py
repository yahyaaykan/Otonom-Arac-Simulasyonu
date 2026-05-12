import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random

# --- KONFİGÜRASYON (V14.0 - 5 ŞERİTLİ AKILLI OTOBAN) ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
LANE_COUNT = 5 # 5 Şeride Çıkarıldı!
LANE_WIDTH = 75 # Şerit genişliği hafif daraltıldı (Ekrana sığması için)
ROAD_WIDTH = LANE_WIDTH * LANE_COUNT
ROAD_X = (SCREEN_WIDTH - ROAD_WIDTH) // 2

CAR_WIDTH = 28
CAR_HEIGHT = 48
MAX_SPEED = 18.0 
MAX_SPEED = 18.0 
MIN_SPEED = 3.0 # Öndeki araca çarpmamak için çok daha yavaşlayabilmesi gerekiyor! (Eskiden 8.0'dı, istese de duramıyordu)
ACCELERATION = 0.5 
BRAKE_FORCE = 1.6 # Acil fren gücü daha da artırıldı
STEERING_SMOOTHING = 0.2

LIDAR_COUNT = 9 # 9 yönlü akıllı tarama (yanlar dahil)
LIDAR_MAX_DIST = 550
FPS = 60

# Renkler
COLOR_BG = (5, 5, 10)
COLOR_ROAD = (20, 20, 30)
COLOR_LINE = (230, 230, 255)
COLOR_AGENT = (0, 255, 255)
COLOR_NPC = (255, 60, 60)
COLOR_SIGNAL = (255, 215, 0)
COLOR_SIDE = (220, 0, 0)

class Entity:
    def __init__(self, x, y, speed, color, id=-1):
        self.x = x
        self.y = y
        self.target_x = x
        self.speed = speed
        self.color = color
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
        self.signal_state = 0
        self.signal_timer = 0
        self.id = id

    def get_rect(self):
        return pygame.Rect(self.x - self.width//2, self.y - self.height//2, self.width, self.height)

    def update_npc(self, agent_speed, others):
        target_speed = agent_speed * 0.65
        if self.speed > target_speed: self.speed -= 0.1
        elif self.speed < target_speed: self.speed += 0.05
        
        my_lane = int((self.x - ROAD_X) // LANE_WIDTH)
        for other in others:
            if other == self: continue
            other_lane = int((other.x - ROAD_X) // LANE_WIDTH)
            if my_lane == other_lane and self.y > other.y:
                dist = self.y - other.y
                if dist < 160:
                    self.speed = max(2.5, other.speed - 0.5)
                    return

class CarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super(CarEnv, self).__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.action_space = spaces.Discrete(4)
        # Gözlem: 9 LiDAR + X + Speed + Left_Free + Right_Free = 13 boyutlu mükemmel gözlem uzayı
        self.observation_space = spaces.Box(low=0, high=1, shape=(LIDAR_COUNT + 4,), dtype=np.float32)
        self.npc_id_counter = 0
        
        # UI İçin Eğitim İstatistikleri
        self.info_episode = 0
        self.info_epsilon = 1.0
        self.info_loss = 0.0
        self.info_action = "BEKLENIYOR..."
        self.action_display_timer = 0
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 5 Şeridin Tam Ortası (LANE_COUNT // 2 = 2)
        start_lane = 2
        start_x = ROAD_X + (start_lane * LANE_WIDTH) + (LANE_WIDTH // 2)
        self.agent = Entity(start_x, SCREEN_HEIGHT - 150, 10.0, COLOR_AGENT)
        self.agent.target_x = self.agent.x
        self.npcs = []
        for _ in range(12):
            self._spawn_npc(y_pos=random.randint(-1600, 50))
        self.steps = 0
        self.total_reward = 0
        self.road_offset = 0
        self.lane_change_cooldown = 0
        self.passed_npcs = set()
        return self._get_obs(), {}

    def _spawn_npc(self, y_pos=-250):
        lane = random.randint(0, LANE_COUNT - 1)
        x = ROAD_X + (lane * LANE_WIDTH) + (LANE_WIDTH // 2)
        
        # --- DUVAR ENGELLEME SİSTEMİ (WALL PREVENTION V14.0) ---
        # Aynı y seviyesindeki araçları say
        same_y_count = 0
        for npc in self.npcs:
            if abs(y_pos - npc.y) < 100:
                same_y_count += 1
            if abs(x - npc.x) < 20 and abs(y_pos - npc.y) < 200:
                return # Aynı şeritte üst üste binme yasak
        
        # Eğer y seviyesinde 3'ten fazla araç varsa, yolu kapatmamak için spawn iptal
        if same_y_count >= 3:
            return 

        # Trafiği daha gerçekçi yapmak için 3 farklı araç tipi
        arac_tipi = random.random()
        if arac_tipi < 0.2:
            speed = random.uniform(5.0, 7.5)   # Yavaş Araç (Kamyon vb.)
        elif arac_tipi < 0.8:
            speed = random.uniform(8.5, 11.5)  # Normal Araç
        else:
            speed = random.uniform(12.5, 15.0) # Hızlı Sollayan Araç
            
        # Eğer ajan henüz çok yavaşsa, en hızlı araçlar bile arkadan füze gibi çarpmasın
        if speed > self.agent.speed + 6.0:
            speed = self.agent.speed + 6.0
        
        self.npc_id_counter += 1
        self.npcs.append(Entity(x, y_pos, speed, COLOR_NPC, id=self.npc_id_counter))

    def step(self, action):
        self.steps += 1
        reward = 0.0

        obs = self._get_obs()
        
        # LiDAR indeksleri: 0=-90(Sol), 1=-45(SolÖn), 4=0(Ön), 7=45(SağÖn), 8=90(Sağ)
        lidar_front = obs[4]
        lidar_sol_on = obs[1]
        lidar_sag_on = obs[7]
        
        # Doğrudan şerit boşluk durumu (0 veya 1)
        left_free = obs[-2] > 0.5
        right_free = obs[-1] > 0.5

        onunde_trafik_var = lidar_front < 0.65
        onunde_cok_yakin_trafik = lidar_front < 0.45

        # UI için son aksiyonu kaydet (Çok hızlı değişmesini önlemek için gecikme)
        if self.action_display_timer <= 0:
            aksiyonlar = ["FREN (YAVASLA)", "GAZ (HIZLAN)", "SOLA GEC (<-)", "SAGA GEC (->)"]
            self.info_action = aksiyonlar[action]
            # Saniyede 60 kare çalışıyor. Yazının 20 kare (~0.3 saniye) ekranda sabit kalmasını sağla
            self.action_display_timer = 20 
        else:
            self.action_display_timer -= 1

        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1

        self.agent.signal_state = 0
        if self.lane_change_cooldown == 0:
            if action in [2, 3]:  # Şerit değiştirme isteği
                if action == 2: # Sola geçiş (Sollama)
                    if not left_free:
                        reward -= 50.0 # Dolu şeride kırmaya çalışmak büyük ceza
                    else:
                        if onunde_trafik_var:
                            reward += 15.0 # Önü dolu, sol açık. Mantıklı güvenli sollama!
                        else:
                            reward -= 20.0 # Önü açıkken gereksiz sola geçiş (Makas Atma Cezası)
                            
                    new_target = max(self.agent.target_x - LANE_WIDTH, ROAD_X + LANE_WIDTH / 2)
                    if new_target != self.agent.target_x and left_free:
                        self.agent.target_x = new_target
                        self.lane_change_cooldown = 45 # Daha yavaş şerit değişimi
                        self.agent.signal_state = 1

                elif action == 3: # Sağa geçiş
                    if not right_free:
                        reward -= 50.0 # Dolu şeride kırmak
                    else:
                        if onunde_trafik_var:
                            reward += 5.0 # Sola geçmek sollama için daha iyi ama sağ da fena değil
                        else:
                            reward -= 20.0 # Gereksiz yere sağa kırma (Zikzak / Makas)

                    new_target = min(self.agent.target_x + LANE_WIDTH, ROAD_X + ROAD_WIDTH - LANE_WIDTH / 2)
                    if new_target != self.agent.target_x and right_free:
                        self.agent.target_x = new_target
                        self.lane_change_cooldown = 45
                        self.agent.signal_state = 2

        self.agent.x += (self.agent.target_x - self.agent.x) * STEERING_SMOOTHING

        if action == 1: # GAZ
            self.agent.speed = min(self.agent.speed + ACCELERATION, MAX_SPEED)
            if onunde_cok_yakin_trafik:
                reward -= 150.0 # ÇOK KRİTİK: Burnunun dibinde araç varken gazlarsan mahvolursun!
            elif onunde_trafik_var:
                reward -= 20.0 # Takip mesafesine girdi, gazlamayı bırakmalı.
                
        elif action == 0: # FREN
            self.agent.speed = max(self.agent.speed - BRAKE_FORCE, MIN_SPEED)
            if onunde_cok_yakin_trafik:
                reward += 50.0 # Acil fren hayat kurtarır! (Mükemmel refleks)
            elif onunde_trafik_var:
                reward += 20.0 # Takip mesafesini korumak için yavaşladı
            else:
                reward -= 20.0 # Önü boşken gereksiz yere fren yaparsa ceza ver!
                
        # Ceza: Çok yakın takip mesafesinde sürünmek (Ödül farmı yapmasını önler)
        if onunde_cok_yakin_trafik:
            reward -= 5.0

        if self.agent.speed < MIN_SPEED:
            self.agent.speed = MIN_SPEED

        scroll_speed = self.agent.speed
        self.road_offset = (self.road_offset + scroll_speed) % 100

        for npc in self.npcs:
            npc.update_npc(self.agent.speed, self.npcs + [self.agent])
            npc.y += (scroll_speed - npc.speed)
            if npc.y > self.agent.y and npc.id not in self.passed_npcs:
                reward += 10.0   # Geçiş ödülü azaltıldı (60->10), amaç yarışmak değil güvenli gitmek
                self.passed_npcs.add(npc.id)

        self.npcs = [n for n in self.npcs if n.y < SCREEN_HEIGHT + 400 and n.y > -2000]
        if len(self.npcs) < 18 and random.random() < 0.28:
            self._spawn_npc(random.randint(-450, -150))

        terminated = False
        agent_r = self.agent.get_rect()
        for npc in self.npcs:
            if agent_r.colliderect(npc.get_rect()):
                reward = -5000   # Kaza cezası tekrar maksimuma çekildi! Çarpmamayı kesin olarak öğrenmeli.
                terminated = True
                break

        # Şerit merkezi ödülü (İyi şoför şeridi ortalar)
        lane_idx = round((self.agent.x - ROAD_X - LANE_WIDTH / 2) / LANE_WIDTH)
        lane_center = ROAD_X + (lane_idx * LANE_WIDTH) + LANE_WIDTH / 2
        dist_c = abs(self.agent.x - lane_center)
        reward += math.cos((dist_c / (LANE_WIDTH / 0.95)) * (math.pi / 2)) * 3.0

        # Optimum Seyir Hızı Ödülü (Cruising Speed)
        # Maksimum hıza ulaşmak yerine güvenli ve stabil bir hızda (örn: 14) kalmayı ödüllendir
        optimum_hiz = 14.0
        if abs(self.agent.speed - optimum_hiz) < 2.0 and not onunde_cok_yakin_trafik:
            reward += 4.0
        
        # Ceza: Eğer önü boşsa ama çok yavaş (kaplumbağa) gidiyorsa
        if lidar_front > 0.8 and self.agent.speed < 10.0:
            reward -= 5.0

        self.total_reward += reward
        truncated = self.steps >= 6000
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        lidar = []
        # Ajan için kusursuz görüş açıları:
        # Negatif açılar SOLA, pozitif açılar SAĞA bakar. 0 derece tam İLERİDİR.
        angles = [-90, -45, -20, -10, 0, 10, 20, 45, 90]
        for a in angles:
            rad = math.radians(a)
            d = self._cast_ray(rad)
            lidar.append(d / LIDAR_MAX_DIST)
            
        rel_x = (self.agent.x - ROAD_X) / ROAD_WIDTH
        speed_norm = self.agent.speed / MAX_SPEED
        
        # === YAN ŞERİT DOLULUK SENSÖRLERİ ===
        my_lane = round((self.agent.x - ROAD_X - LANE_WIDTH / 2) / LANE_WIDTH)
        left_free = 1.0 if my_lane > 0 else 0.0
        right_free = 1.0 if my_lane < LANE_COUNT - 1 else 0.0
        
        agent_r = self.agent.get_rect()
        for npc in self.npcs:
            npc_lane = round((npc.x - ROAD_X - LANE_WIDTH / 2) / LANE_WIDTH)
            # Eğer NPC y ekseninde yakınımızdaysa (çarpışma riski varsa)
            if abs(npc.y - self.agent.y) < 140:
                if npc_lane == my_lane - 1: left_free = 0.0
                if npc_lane == my_lane + 1: right_free = 0.0

        return np.array(lidar + [rel_x, speed_norm, left_free, right_free], dtype=np.float32)

    def _cast_ray(self, rad):
        sx, sy = self.agent.x, self.agent.y
        # dx: negatifse sol (-x), pozitifse sağ (+x)
        dx = math.sin(rad)
        # dy: yukarı doğru gitmek istiyoruz (Pygame'de yukarı = eksi y). 0 açısında dx=0, dy=-1 olmalı.
        dy = -math.cos(rad)
        
        for d in range(10, LIDAR_MAX_DIST, 20):
            tx, ty = sx + dx*d, sy + dy*d
            if tx < ROAD_X or tx > ROAD_X + ROAD_WIDTH: return d
            for npc in self.npcs:
                if npc.get_rect().collidepoint(tx, ty): return d
        return LIDAR_MAX_DIST

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("AI 5-Lane Intelligent Highway V14.0")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        if self.screen is None: return

        self.screen.fill(COLOR_BG)
        # 5 Şeritli Yol
        pygame.draw.rect(self.screen, COLOR_ROAD, (ROAD_X, 0, ROAD_WIDTH, SCREEN_HEIGHT))
        
        # Kenar Bariyerler
        for y in range(int(self.road_offset*4) - 100, SCREEN_HEIGHT+100, 40):
            color = (255, 255, 255) if (y // 40) % 2 == 0 else COLOR_SIDE
            pygame.draw.rect(self.screen, color, (ROAD_X - 12, y, 12, 40))
            pygame.draw.rect(self.screen, color, (ROAD_X + ROAD_WIDTH, y, 12, 40))

        # 5 Şerit Çizgileri
        for i in range(1, LANE_COUNT):
            x = ROAD_X + i * LANE_WIDTH
            for y in range(int(self.road_offset) - 100, SCREEN_HEIGHT+100, 100):
                pygame.draw.line(self.screen, COLOR_LINE, (x, y), (x, y+75), 3)

        for ent in self.npcs + [self.agent]:
            r = ent.get_rect()
            pygame.draw.rect(self.screen, ent.color, r, border_radius=12)
            pygame.draw.rect(self.screen, (255, 255, 255), r, 2, border_radius=12)
            
            if hasattr(ent, "signal_state") and ent.signal_state > 0:
                self.agent.signal_timer = (self.agent.signal_timer + 1) % 10
                if self.agent.signal_timer < 5:
                    if ent.signal_state == 1: pygame.draw.circle(self.screen, COLOR_SIGNAL, (int(ent.x - 14), int(ent.y + 24)), 9)
                    elif ent.signal_state == 2: pygame.draw.circle(self.screen, COLOR_SIGNAL, (int(ent.x + 14), int(ent.y + 24)), 9)

        # 5-LANE HUD & EĞİTİM BİLGİLERİ (GELİŞTİRİLMİŞ TASARIM)
        s_hud = pygame.Surface((SCREEN_WIDTH, 140), pygame.SRCALPHA)
        s_hud.fill((15, 15, 20, 240)) # Daha koyu transparan arkaplan
        self.screen.blit(s_hud, (0, 0))
        
        # HUD Alt Sınır Çizgisi (Neon Mavi)
        pygame.draw.line(self.screen, (0, 200, 255), (0, 138), (SCREEN_WIDTH, 138), 3)
        
        f_val = pygame.font.SysFont("Impact", 42)
        f_sub = pygame.font.SysFont("Arial Black", 14)
        f_info = pygame.font.SysFont("Consolas", 18, bold=True)
        f_action = pygame.font.SysFont("Arial Black", 22)
        
        # Sol taraf: Hız ve Skor
        self.screen.blit(f_val.render(f"{self.agent.speed*12:.0f} KM/H", True, (0, 255, 255)), (20, 10))
        self.screen.blit(f_sub.render(f"TOPLAM ODUL: {self.total_reward:.0f}", True, (255, 255, 255)), (20, 65))
        self.screen.blit(f_sub.render(f"Sollanan Arac: {len(self.passed_npcs)}", True, (200, 200, 200)), (20, 85))
        
        # Sağ taraf: Eğitim Durumu (Daha hizalı)
        y_pos = 15
        self.screen.blit(f_info.render(f"BOLUM   : {self.info_episode}", True, (255, 200, 0)), (340, y_pos))
        self.screen.blit(f_info.render(f"EPSILON : {self.info_epsilon:.3f}", True, (255, 100, 100)), (340, y_pos+25))
        self.screen.blit(f_info.render(f"KAYIP   : {self.info_loss:.2f}", True, (100, 255, 100)), (340, y_pos+50))
        
        # Alt taraf: Anlık Aksiyon Kararı (Daha büyük ve sabit okunabilir)
        aksiyon_renk = (200, 200, 255)
        if "FREN" in self.info_action: aksiyon_renk = (255, 80, 80)
        elif "GAZ" in self.info_action: aksiyon_renk = (80, 255, 80)
        elif "SOLA" in self.info_action or "SAGA" in self.info_action: aksiyon_renk = (255, 215, 0) # Altın sarısı
        
        # Arkaplan kutusu aksiyon için
        pygame.draw.rect(self.screen, (30, 30, 40), (20, 105, 300, 30), border_radius=5)
        self.screen.blit(f_action.render(f"KARAR: {self.info_action}", True, aksiyon_renk), (25, 105))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
