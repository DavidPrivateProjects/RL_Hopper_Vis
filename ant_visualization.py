import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

# Window dimensions
WIDTH, HEIGHT = 1200, 800

# Data structures for the ant model
joints = {} # Joint name to position mapping
limbs = [] # List of (start_joint, end_joint) tuples
click_counts = {} # Number of clicks per limb (for coloring)

# Mouse and rotation state
rotation = [0, 0]
mouse_down = False
last_mouse_pos = (0, 0)

# UI buttons using pygame.Rect (for logical interaction only)
reset_button = pygame.Rect(1000, 20, 150, 40)
video_button = pygame.Rect(1000, 70, 150, 40)

# Font initialization for UI text
pygame.font.init()
font = pygame.font.SysFont("Arial", 20)

# Initialize Pygame, OpenGL settings, and projection matrix
def init():
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.65, 0.75, 0.85, 1.0)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

# Construct the 3D ant structure from joint positions and limb connections
def build_ant_structure():
    joints.clear()
    limbs.clear()
    click_counts.clear()

    torso = np.array([0, 1, 0])
    joints["torso"] = torso

    shin_length = 1.5
    leg_offsets = {
        "front_left": [0.5, 0.0, 0.4],
        "front_right": [-0.5, 0.0, 0.4],
        "back_left": [0.5, 0.0, -0.4],
        "back_right": [-0.5, 0.0, -0.4],
    }

    for leg, offset in leg_offsets.items():
        hip = torso + np.array(offset)
        joints[f"{leg}_hip"] = hip

        knee = hip + np.array([offset[0]*2, 0, offset[2]*2])
        joints[f"{leg}_knee"] = knee

        foot = knee + np.array([offset[0], -shin_length, offset[2]])
        joints[f"{leg}_foot"] = foot

        limbs.append(("torso", f"{leg}_hip"))
        limbs.append((f"{leg}_hip", f"{leg}_knee"))
        limbs.append((f"{leg}_knee", f"{leg}_foot"))

# Render a joint as a sphere
def draw_sphere(pos, radius=2, color=(0, 0, 0)):
    glPushMatrix()
    glTranslatef(*pos)
    glColor3fv(color)
    quad = gluNewQuadric()
    gluSphere(quad, radius, 20, 20)
    glPopMatrix()

# Render a limb as a cylinder between two joints
def draw_limb(start, end, radius=0.25, colors=(1, 1, 1)):
    start = np.array(start)
    end = np.array(end)
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        return

    direction /= length
    up = np.array([0, 0, 1])
    axis = np.cross(up, direction)
    angle = math.degrees(math.acos(np.clip(np.dot(up, direction), -1.0, 1.0)))

    glPushMatrix()
    glTranslatef(*start)

    if np.linalg.norm(axis) > 1e-6:
        glRotatef(angle, *axis)

    glColor3f(*colors)
    quad = gluNewQuadric()
    gluCylinder(quad, radius, radius, length, 20, 1)
    glPopMatrix()

# Render the entire ant model
def render_ant():
    RADIUS_TORSO = 0.8
    RADIUS_LEGS = 0.3

    for joint, pos in joints.items():
        draw_sphere(pos, RADIUS_LEGS if "torso" not in joint else RADIUS_TORSO, (0, 0, 0))

    for i, (start, end) in enumerate(limbs):
        count = click_counts.get(i, 0)
        intensity = min(1.0, count * 0.25)
        color = (1.0, 1.0 - intensity, 1.0 - intensity)
        draw_limb(joints[start], joints[end], RADIUS_LEGS, color)

# Convert 2D mouse position to 3D picking ray
def get_mouse_ray(mx, my):
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)

    near = gluUnProject(mx, HEIGHT - my, 0.0, modelview, projection, viewport)
    far = gluUnProject(mx, HEIGHT - my, 1.0, modelview, projection, viewport)

    ray_origin = np.array(near)
    ray_dir = np.array(far) - ray_origin
    ray_dir /= np.linalg.norm(ray_dir)
    return ray_origin, ray_dir

# Ray-cylinder intersection test for limb selection
def ray_cylinder_intersect(ray_origin, ray_dir, p1, p2, radius):
    d = p2 - p1
    m = ray_origin - p1
    n = ray_dir

    d_norm = d / np.linalg.norm(d)
    m_proj = m - np.dot(m, d_norm) * d_norm
    n_proj = n - np.dot(n, d_norm) * d_norm

    a = np.dot(n_proj, n_proj)
    b = 2 * np.dot(n_proj, m_proj)
    c = np.dot(m_proj, m_proj) - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0.0:
        return False

    sqrt_disc = np.sqrt(disc)
    for t in [(-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)]:
        if t < 0:
            continue
        hit = ray_origin + t * ray_dir
        if 0 <= np.dot(hit - p1, d_norm) <= np.linalg.norm(d):
            return True
    return False

# Handle mouse clicks for selecting limbs
def handle_click(mx, my):
    ray_origin, ray_dir = get_mouse_ray(mx, my)
    for i, (start, end) in enumerate(limbs):
        if ray_cylinder_intersect(ray_origin, ray_dir, joints[start], joints[end], 0.3):
            click_counts[i] = click_counts.get(i, 0) + 1
            break

# Update camera rotation based on mouse movement
def handle_mouse_motion(pos):
    global last_mouse_pos, rotation
    if mouse_down:
        dx = pos[0] - last_mouse_pos[0]
        dy = pos[1] - last_mouse_pos[1]
        rotation[0] += dx * 0.4
        rotation[1] += dy * 0.4
    last_mouse_pos = pos

# Reset all limb click counts
def reset_clicks():
    for joint in click_counts:
        click_counts[joint] = 0

# Toggle video (automatic rotation)
def play_video(current_state):
    return not current_state

# Draw a filled rectangle (used for buttons)
def draw_rect(x, y, w, h):
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()

# Draw text using Pygame font rendering
def draw_text(x, y, text, font, color=(255, 255, 255),
             surface_color=(0, 0, 0, 0)):
    text_surface = font.render(text, True, surface_color, color)
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    width, height = text_surface.get_size()

    glRasterPos2f(x, y)
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

# Draw the UI buttons
def draw_buttons():
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)

    # Reset Button
    glColor3f(0.8, 0.2, 0.2)
    draw_rect(reset_button.x, reset_button.y, reset_button.width, reset_button.height)
    draw_text(reset_button.x+19, reset_button.y+32, 
              "Reset Settings", font, (204, 51, 51, 0))

    # Rotate (video) button
    glColor3f(0.2, 0.6, 0.2)
    draw_rect(video_button.x, video_button.y, video_button.width, video_button.height)
    draw_text(video_button.x+13, video_button.y+32, 
              "Rotate", font,
              (51, 153, 51, 0))

    glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def main():
    global mouse_down
    init()
    build_ant_structure()
    clock = pygame.time.Clock()
    video_bool = False

    while True:
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                mouse_down = True
                if reset_button.collidepoint(mx, my):
                    reset_clicks()
                elif video_button.collidepoint(mx, my):
                    video_bool = play_video(video_bool)
                else:
                    handle_click(mx, my)
            elif event.type == MOUSEBUTTONUP and event.button == 1:
                mouse_down = False
            elif event.type == MOUSEMOTION:
                handle_mouse_motion(event.pos)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, -1, -10)
        glRotatef(rotation[1], 1, 0, 0)
        glRotatef(rotation[0], 0, 1, 0)

        if video_bool:
            rotation[0] += 1 # Continuous rotation when video mode is active
        
        render_ant()
        draw_buttons()

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
