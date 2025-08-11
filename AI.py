# 2D-Fußball – Zwei lernende AIs (DQN) – **5x parallel**, **Resume**, **Zuschauen**, **sicher abbrechen**
# ======================================================================================
# Alles in EINER Datei. Standardmäßig werden 5 Spiele gleichzeitig trainiert.
# Du kannst jederzeit mit STRG+C abbrechen – es wird automatisch gespeichert.
# Mit --resume kannst du später nahtlos weitermachen. Zuschauen während des Trainings mit --render-every.
#
# Installation (einmalig):
#   pip install torch pygame numpy
#
# Beispiele:
#   python AI.py --episodes 1000                    # 5 parallele Spiele, trainieren und speichern
#   python AI.py --episodes 1000 --resume           # vom letzten Stand weitertrainieren
#   python AI.py --episodes 2000 --render-every 50  # jede 50. Episode Env #0 anzeigen
#   python AI.py --episodes 1000 --num-envs 8       # 8 parallele Spiele
#   python AI.py --play                              # nur zuschauen (greedy, geladene Modelle)

import math
import random
import argparse
from collections import deque, namedtuple
import os

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# ENV-Konstanten
# -----------------------------
WIDTH, HEIGHT = 900, 540
FIELD_MARGIN = 40
GOAL_WIDTH = 160
BALL_RADIUS = 8
PLAYER_RADIUS = 12

BALL_FRICTION = 0.992
PLAYER_FRICTION = 0.90
ACCEL = 0.9
MAX_PLAYER_SPEED = 5.0
KICK_FORCE = 10.0

MAX_STEPS = 2000  # max Schritte pro Spiel (Episode)

# Farben
GREEN = (30, 140, 70)
DARK_GREEN = (18, 110, 56)
WHITE = (240, 240, 240)
LINE = (225, 245, 225)
RED = (220, 70, 70)
BLUE = (70, 120, 220)
YELLOW = (240, 220, 80)

# -----------------------------
# Hilfsfunktionen
# -----------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def vec_len(x, y):
    return math.hypot(x, y)

# -----------------------------
# Environment (Gym-ähnlich)
# -----------------------------
class SoccerEnv:
    """1v1 Soccer. step(aL, aR) -> obs, (rL, rR), done, info
    State (12): [bx,by,bvx,bvy, Lx,Ly,Lvx,Lvy, Rx,Ry,Rvx,Rvy] – normalisiert
    Actions (6): 0 noop, 1 up, 2 down, 3 left, 4 right, 5 kick
    Rewards: +goal_reward bei eigenem Tor, -goal_reward bei Gegentor, kleine shaping-Rewards * shape_scale
    """
    def __init__(self, render=False, end_on_goal=False, goal_reward=1.0, shape_scale=1.0):
        self.render_enabled = render
        self.screen = None
        self.clock = None
        self.end_on_goal = end_on_goal
        self.goal_reward = goal_reward
        self.shape_scale = shape_scale
        # Goal-Score (persistiert über Episoden hinweg, bis Programmende oder Reset)
        self.scoreL = 0
        self.scoreR = 0
        self.font = None
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("RL Soccer 1v1")
            self.font = pygame.font.SysFont("Arial", 22)
        self.reset()

    def reset(self):
        self.left = {
            'x': FIELD_MARGIN+120.0,
            'y': HEIGHT/2,
            'vx': 0.0,
            'vy': 0.0,
        }
        self.right = {
            'x': WIDTH-FIELD_MARGIN-120.0,
            'y': HEIGHT/2,
            'vx': 0.0,
            'vy': 0.0,
        }
        self.ball = {
            'x': WIDTH/2,
            'y': HEIGHT/2,
            'vx': random.uniform(-1.5, 1.5),
            'vy': random.uniform(-1.0, 1.0),
        }
        self.steps = 0
        return self._obs()

    def _center_ball(self):
        # Ball zurück in die Mitte (Anstoß), Spieler bleiben wo sie sind
        self.ball['x'] = WIDTH/2
        self.ball['y'] = HEIGHT/2
        self.ball['vx'] = random.uniform(-1.5, 1.5)
        self.ball['vy'] = random.uniform(-1.0, 1.0)

    # -------------------------
    # Physics + Step
    # -------------------------
    def step(self, aL, aR):
        self.steps += 1
        # apply actions -> velocities
        self._apply_action(self.left, aL)
        self._apply_action(self.right, aR)
        self._update_players()
        self._update_ball()
        self._resolve_collisions()
        # rewards
        goal = self._check_goal()
        rL = 0.0
        rR = 0.0
        if goal == 'L':
            rL = self.goal_reward; rR = -self.goal_reward
            # Score hochzählen (Linkes Team hat getroffen)
            self.scoreL += 1
            # Ball sofort wieder in die Mitte, wenn wir nicht bei Tor beenden
            if not self.end_on_goal:
                self._center_ball()
        elif goal == 'R':
            rL = -self.goal_reward; rR = self.goal_reward
            # Rechtes Team hat getroffen
            self.scoreR += 1
            if not self.end_on_goal:
                self._center_ball()
        else:
            # shaping: Nähe zum Ball + Ballbewegung Richtung Gegnertor
            rL += self._shaping(side='L')
            rR += self._shaping(side='R')
        done = (goal is not None and self.end_on_goal) or self.steps >= MAX_STEPS
        if self.render_enabled:
            self.render()
        return self._obs(), (rL, rR), done, {"goal": goal}

    def _apply_action(self, p, a):
        if a == 1:   # up
            p['vy'] -= ACCEL
        elif a == 2: # down
            p['vy'] += ACCEL
        elif a == 3: # left
            p['vx'] -= ACCEL
        elif a == 4: # right
            p['vx'] += ACCEL
        elif a == 5: # kick
            # if close to ball -> impulse towards opponent goal
            dx = self.ball['x'] - p['x']
            dy = self.ball['y'] - p['y']
            d = vec_len(dx, dy)
            if d < PLAYER_RADIUS + BALL_RADIUS + 3:
                goal_x = FIELD_MARGIN if p is self.right else WIDTH - FIELD_MARGIN
                gx, gy = goal_x, HEIGHT/2
                ux = (gx - self.ball['x'])
                uy = (gy - self.ball['y'])
                ul = vec_len(ux, uy) or 1.0
                self.ball['vx'] += (ux/ul) * KICK_FORCE
                self.ball['vy'] += (uy/ul) * KICK_FORCE
        # speed cap
        sp = vec_len(p['vx'], p['vy'])
        if sp > MAX_PLAYER_SPEED:
            p['vx'] *= MAX_PLAYER_SPEED / sp
            p['vy'] *= MAX_PLAYER_SPEED / sp

    def _update_players(self):
        for p in (self.left, self.right):
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vx'] *= PLAYER_FRICTION
            p['vy'] *= PLAYER_FRICTION
            # boundaries
            p['x'] = clamp(p['x'], FIELD_MARGIN+PLAYER_RADIUS, WIDTH-FIELD_MARGIN-PLAYER_RADIUS)
            p['y'] = clamp(p['y'], FIELD_MARGIN+PLAYER_RADIUS, HEIGHT-FIELD_MARGIN-PLAYER_RADIUS)

    def _update_ball(self):
        b = self.ball
        b['x'] += b['vx']
        b['y'] += b['vy']
        b['vx'] *= BALL_FRICTION
        b['vy'] *= BALL_FRICTION
        # top/bottom bounce
        top = FIELD_MARGIN; bot = HEIGHT - FIELD_MARGIN
        if b['y'] - BALL_RADIUS < top:
            b['y'] = top + BALL_RADIUS; b['vy'] *= -0.9
        if b['y'] + BALL_RADIUS > bot:
            b['y'] = bot - BALL_RADIUS; b['vy'] *= -0.9
        # left/right with goals opening
        left = FIELD_MARGIN; right = WIDTH - FIELD_MARGIN
        gt = HEIGHT/2 - GOAL_WIDTH/2; gb = HEIGHT/2 + GOAL_WIDTH/2
        if b['x'] - BALL_RADIUS < left:
            if not (gt < b['y'] < gb):
                b['x'] = left + BALL_RADIUS; b['vx'] *= -0.9
        if b['x'] + BALL_RADIUS > right:
            if not (gt < b['y'] < gb):
                b['x'] = right - BALL_RADIUS; b['vx'] *= -0.9

    def _resolve_collisions(self):
        # player-ball
        for p in (self.left, self.right):
            dx = self.ball['x'] - p['x']
            dy = self.ball['y'] - p['y']
            d = vec_len(dx, dy)
            over = PLAYER_RADIUS + BALL_RADIUS - d
            if over > 0:
                nx, ny = (dx/d, dy/d) if d else (1.0, 0.0)
                self.ball['x'] += nx * over
                self.ball['y'] += ny * over
                self.ball['vx'] += p['vx'] * 0.15
                self.ball['vy'] += p['vy'] * 0.15
        # players
        dx = self.right['x'] - self.left['x']
        dy = self.right['y'] - self.left['y']
        d = vec_len(dx, dy)
        over = 2*PLAYER_RADIUS - d
        if over > 0:
            nx, ny = (dx/d, dy/d) if d else (1.0, 0.0)
            self.left['x'] -= nx * over/2
            self.left['y'] -= ny * over/2
            self.right['x'] += nx * over/2
            self.right['y'] += ny * over/2

    def _check_goal(self):
        b = self.ball
        left = FIELD_MARGIN; right = WIDTH - FIELD_MARGIN
        gt = HEIGHT/2 - GOAL_WIDTH/2; gb = HEIGHT/2 + GOAL_WIDTH/2
        if gt < b['y'] < gb:
            if b['x'] - BALL_RADIUS <= left:
                return 'R'
            if b['x'] + BALL_RADIUS >= right:
                return 'L'
        return None

    def _shaping(self, side='L'):
        # Normalisierte, kleine Shaping-Rewards (werden mit shape_scale gewichtet)
        # 1) Nähe zum Ball (0..~0.01)
        if side == 'L':
            px, py = self.left['x'], self.left['y']
            goal_x = WIDTH - FIELD_MARGIN
        else:
            px, py = self.right['x'], self.right['y']
            goal_x = FIELD_MARGIN
        d = vec_len(self.ball['x']-px, self.ball['y']-py) + 1e-6
        proximity = 0.01 * (1.0 / d)  # kleiner Reward, je näher am Ball

        # 2) Ballrichtung zum Gegner (stark begrenzt & normalisiert)
        nx = (goal_x - self.ball['x']) / WIDTH           # ~[-1..1]
        nvx = max(-1.0, min(1.0, self.ball['vx'] / 8.0)) # ~[-1..1]
        ball_dir = 0.02 * (nx * nvx)

        # 3) Mini-Zeitstrafe, damit schneller abgeschlossen wird
        time_penalty = -0.0002
        return self.shape_scale * (proximity + ball_dir + time_penalty)

    # -------------------------
    # Rendering
    # -------------------------
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            # Score-Reset per Taste 'R'
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.scoreL = 0
                self.scoreR = 0
        surf = self.screen
        if self.font is None:
            # Falls Render erst später aktiviert wird
            self.font = pygame.font.SysFont("Arial", 22)
        surf.fill(GREEN)
        pygame.draw.rect(surf, DARK_GREEN, (FIELD_MARGIN, FIELD_MARGIN, WIDTH-2*FIELD_MARGIN, HEIGHT-2*FIELD_MARGIN), border_radius=14)
        # midfield
        pygame.draw.line(surf, LINE, (WIDTH//2, FIELD_MARGIN), (WIDTH//2, HEIGHT-FIELD_MARGIN), 2)
        pygame.draw.circle(surf, LINE, (WIDTH//2, HEIGHT//2), 60, 2)
        # goals
        gt = HEIGHT//2 - GOAL_WIDTH//2; gb = HEIGHT//2 + GOAL_WIDTH//2
        pygame.draw.rect(surf, WHITE, (FIELD_MARGIN-5, gt, 8, GOAL_WIDTH))
        pygame.draw.rect(surf, WHITE, (WIDTH-FIELD_MARGIN-8, gt, 8, GOAL_WIDTH))
        # ball
        pygame.draw.circle(surf, YELLOW, (int(self.ball['x']), int(self.ball['y'])), BALL_RADIUS)
        # players
        pygame.draw.circle(surf, BLUE, (int(self.left['x']), int(self.left['y'])), PLAYER_RADIUS)
        pygame.draw.circle(surf, RED, (int(self.right['x']), int(self.right['y'])), PLAYER_RADIUS)
        # HUD: Score
        score_text = f"Blau: {self.scoreL}   Rot: {self.scoreR}   (R = Reset Score)"
        txt = self.font.render(score_text, True, WHITE)
        surf.blit(txt, (WIDTH//2 - txt.get_width()//2, 8))
        pygame.display.flip()
        if self.clock:
            self.clock.tick(60)

    # -------------------------
    # Observations
    # -------------------------
    def _obs(self):
        # normalize to 0..1 ranges roughly
        def nx(x): return (x - 0) / float(WIDTH)
        def ny(y): return (y - 0) / float(HEIGHT)
        def nvx(vx): return (vx + 8.0) / 16.0
        def nvy(vy): return (vy + 8.0) / 16.0
        o = [
            nx(self.ball['x']), ny(self.ball['y']), nvx(self.ball['vx']), nvy(self.ball['vy']),
            nx(self.left['x']), ny(self.left['y']), nvx(self.left['vx']), nvy(self.left['vy']),
            nx(self.right['x']), ny(self.right['y']), nvx(self.right['vx']), nvy(self.right['vy'])
        ]
        return np.array(o, dtype=np.float32)

# -----------------------------
# DQN-Agent
# -----------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        return [self.buf[i] for i in idx]
    def __len__(self):
        return len(self.buf)

class QNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99, device='cpu', memory_capacity=200_000, batch_size=256):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.policy = QNet(state_dim, n_actions).to(device)
        self.target = QNet(state_dim, n_actions).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=memory_capacity)
        self.loss_fn = nn.SmoothL1Loss()
        self.batch_size = batch_size

    def act(self, state, eps):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy(s)
            return int(torch.argmax(q, dim=1).item())

    def remember(self, *args):
        self.memory.push(*args)

    def update(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return 0.0
        batch = self.memory.sample(batch_size)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_vals = self.policy(states).gather(1, actions)
        with torch.no_grad():
            max_next = self.target(next_states).max(1, keepdim=True)[0]
            targets = rewards + (1 - dones) * self.gamma * max_next
        loss = self.loss_fn(q_vals, targets)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optim.step()
        return float(loss.item())

    def soft_update(self, tau=0.01):
        with torch.no_grad():
            for tp, sp in zip(self.target.parameters(), self.policy.parameters()):
                tp.data.mul_(1 - tau).add_(sp.data * tau)

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.policy.state_dict())

# -----------------------------
# Evaluation (gegen Zufall) – Fortschritt sichtbar machen
# -----------------------------

def evaluate_against_random(agent_left, agent_right, matches=20):
    def run(agentL, agentR, play_left=True, games=10):
        wins = 0
        for _ in range(games):
            env = SoccerEnv(render=False)
            state = env.reset()
            done = False
            while not done:
                if play_left:
                    aL = agentL.act(state, eps=0.0)
                    aR = random.randrange(6)
                else:
                    aL = random.randrange(6)
                    aR = agentR.act(state, eps=0.0)
                state, _, done, info = env.step(aL, aR)
            if play_left and env.scoreL > env.scoreR:
                wins += 1
            if (not play_left) and env.scoreR > env.scoreL:
                wins += 1
        return wins
    wl = run(agent_left, agent_right, True, matches//2)
    wr = run(agent_left, agent_right, False, matches//2)
    return wl, wr, matches

# -----------------------------
# Training / Play (mit Resume + parallelen Envs)
# -----------------------------

def train(episodes=1000, render_every=0, save_prefix="soccer_dqn", resume=False, num_envs=5, checkpoint_every=50, end_on_goal=False, batch_size=256, buffer_capacity=200_000, goal_reward=1.0, shape_scale=1.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    envs = [SoccerEnv(render=False, end_on_goal=end_on_goal, goal_reward=goal_reward, shape_scale=shape_scale) for _ in range(num_envs)]

    s_dim = 12
    n_actions = 6
    left_agent = DQNAgent(s_dim, n_actions, lr=1e-3, gamma=0.99, device=device, memory_capacity=buffer_capacity, batch_size=batch_size)
    right_agent = DQNAgent(s_dim, n_actions, lr=1e-3, gamma=0.99, device=device, memory_capacity=buffer_capacity, batch_size=batch_size)

    if resume:
        try:
            left_agent.load(f"{save_prefix}_L.pt")
            right_agent.load(f"{save_prefix}_R.pt")
            print("Modelle geladen – Training wird fortgesetzt.")
        except FileNotFoundError:
            print("Keine gespeicherten Modelle gefunden – starte von Null.")

    eps_start, eps_end, eps_decay = 1.0, 0.05, 0.0004
    eps = eps_start

    print(f"Training on {device} for {episodes} episodes with {num_envs} parallel envs…")
    total_steps = 0

    try:
        for ep in range(1, episodes+1):
            # Episode = alle num_envs Envs werden zurückgesetzt und bis zum Done gespielt
            states = [env.reset() for env in envs]
            dones = [False] * num_envs
            ep_ret_L = np.zeros(num_envs, dtype=np.float32)
            ep_ret_R = np.zeros(num_envs, dtype=np.float32)
            steps_in_ep = 0

            # Render-Umschaltung je Episode für Env #0
            e0 = envs[0]
            want_render = bool(render_every and (ep % render_every == 0))
            if want_render and e0.screen is None:
                pygame.init(); e0.screen = pygame.display.set_mode((WIDTH, HEIGHT)); e0.clock = pygame.time.Clock()
                e0.font = pygame.font.SysFont("Arial", 22)
            # Nur in den gewünschten Episoden rendern
            e0.render_enabled = want_render

            while not all(dones):
                # Falls ein Fenster offen ist, aber Render gerade AUS ist: Events pumpen, damit Windows nicht "keine Rückmeldung" zeigt
                if e0.screen is not None and not e0.render_enabled:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                    if e0.clock:
                        e0.clock.tick(15)  # kleine Pause hält das Fenster responsiv
                # epsilon pro globalem Schritt decayn
                eps = max(eps_end, eps - eps_decay)

                actions_L = [0]*num_envs
                actions_R = [0]*num_envs
                for i in range(num_envs):
                    if not dones[i]:
                        actions_L[i] = left_agent.act(states[i], eps)
                        actions_R[i] = right_agent.act(states[i], eps)

                next_states = [None]*num_envs
                for i in range(num_envs):
                    if dones[i]:
                        continue
                    ns, (rL, rR), done, _ = envs[i].step(actions_L[i], actions_R[i])
                    left_agent.remember(states[i], actions_L[i], rL, ns, float(done))
                    right_agent.remember(states[i], actions_R[i], rR, ns, float(done))
                    next_states[i] = ns
                    ep_ret_L[i] += rL
                    ep_ret_R[i] += rR
                    dones[i] = done

                # Optimierung (einmal pro globalem Schritt)
                left_agent.update()
                right_agent.update()
                left_agent.soft_update(0.01)
                right_agent.soft_update(0.01)

                # States fortschreiben
                for i in range(num_envs):
                    if not dones[i] and next_states[i] is not None:
                        states[i] = next_states[i]

                # Rendering passiert automatisch in env.step(), wenn e0.render_enabled True ist.

                steps_in_ep += 1
                total_steps += 1

            # Speichern & Logging
            if ep % checkpoint_every == 0:
                left_agent.save(f"{save_prefix}_L.pt")
                right_agent.save(f"{save_prefix}_R.pt")
            if ep % 10 == 0:
                print(
                    f"Ep {ep:5d} | envs {num_envs} | steps {steps_in_ep:4d} | "
                    f"return L avg {ep_ret_L.mean():+.3f} R avg {ep_ret_R.mean():+.3f} | eps {eps:.3f}"
                )
            # Kleine Evaluation alle 100 Episoden (gegen Zufall)
            if ep % 100 == 0:
                wl, wr, m = evaluate_against_random(left_agent, right_agent, matches=20)
                print(f"Eval vs Random -> Left wins {wl}/{m//2}, Right wins {wr}/{m//2}")

    except KeyboardInterrupt:
        # Sicher speichern beim Abbruch
        left_agent.save(f"{save_prefix}_L.pt")
        right_agent.save(f"{save_prefix}_R.pt")
        print("Training abgebrochen – Modelle gespeichert.")
        return

    # Finale Speicherung
    left_agent.save(f"{save_prefix}_L.pt")
    right_agent.save(f"{save_prefix}_R.pt")
    print("Training done. Models saved.")


def play(render_fps=60, load_prefix="soccer_dqn", end_on_goal=False, goal_reward=1.0, shape_scale=1.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = SoccerEnv(render=True, end_on_goal=end_on_goal, goal_reward=goal_reward, shape_scale=shape_scale)
    s_dim = 12
    n_actions = 6
    left_agent = DQNAgent(s_dim, n_actions, device=device)
    right_agent = DQNAgent(s_dim, n_actions, device=device)
    try:
        left_agent.load(f"{load_prefix}_L.pt")
        right_agent.load(f"{load_prefix}_R.pt")
        print("Models loaded.")
    except FileNotFoundError:
        print("Keine gespeicherten Modelle gefunden – starte mit untrainierten Gewichten.")

    clock = pygame.time.Clock()
    while True:
        state = env.reset()
        done = False
        while not done:
            aL = left_agent.act(state, eps=0.0)
            aR = right_agent.act(state, eps=0.0)
            state, _, done, _ = env.step(aL, aR)
            env.render()
            clock.tick(render_fps)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help='Nur abspielen (Modelle laden)')
    parser.add_argument('--episodes', type=int, default=1000, help='Trainings-Episoden (jede Episode = Batch aus parallelen Spielen)')
    parser.add_argument('--render-every', type=int, default=0, help='Alle N Episoden Env #0 rendern (0=nie)')
    parser.add_argument('--prefix', type=str, default='soccer_dqn', help='Prefix für Modelldateien')
    parser.add_argument('--resume', action='store_true', help='Vom letzten Stand weitertrainieren')
    parser.add_argument('--num-envs', type=int, default=6, help='Anzahl paralleler Umgebungen (Default: 6 – Auto-Profil)')
    parser.add_argument('--checkpoint-every', type=int, default=50, help='Alle N Episoden automatisch speichern')
    parser.add_argument('--end-on-goal', action='store_true', help='Episode endet bei Tor (statt erst bei max-steps)')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximale Schritte pro Episode (Default: 1000 – Auto-Profil)')
    parser.add_argument('--batch-size', type=int, default=192, help='Batch-Größe für DQN-Updates (Default: 192 – Auto-Profil)')
    parser.add_argument('--buffer-capacity', type=int, default=150000, help='Replay-Buffer-Kapazität (Default: 150000 – Auto-Profil)')
    parser.add_argument('--goal-reward', type=float, default=10.0, help='Reward-Höhe pro Tor (Default: 10.0)')
    parser.add_argument('--shape-scale', type=float, default=1.0, help='Faktor für Shaping-Rewards (Default: 1.0)')
    args = parser.parse_args()

    if args.play:
        MAX_STEPS = args.max_steps
        play(load_prefix=args.prefix, end_on_goal=args.end_on_goal, goal_reward=args.goal_reward, shape_scale=args.shape_scale)
    else:
        MAX_STEPS = args.max_steps
        train(episodes=args.episodes,
              render_every=args.render_every,
              save_prefix=args.prefix,
              resume=args.resume,
              num_envs=args.num_envs,
              checkpoint_every=args.checkpoint_every,
              end_on_goal=args.end_on_goal,
              batch_size=args.batch_size,
              buffer_capacity=args.buffer_capacity,
              goal_reward=args.goal_reward,
              shape_scale=args.shape_scale)

