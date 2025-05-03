import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

WIDTH, HEIGHT = 1200, 800

joints = {}
limbs = []
click_counts = {}
rotation = [0, 0]
mouse_down = False
last_mouse_pos = (0, 0)

def init():
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def build_ant_structure():
    joints.clear()
    limbs.clear()
    click_counts.clear()

    torso = np.array([0, 1, 0])
    joints["torso"] = torso

    # Leg params
    shin_length = 1.5

    leg_offsets = {
        "front_left":  [0.5, 0.0,  0.4],
        "front_right": [-0.5, 0.0,  0.4],
        "back_left":   [0.5, 0.0, -0.4],
        "back_right":  [-0.5, 0.0, -0.4],
    }

    for leg, offset in leg_offsets.items():
        hip = torso + np.array(offset)
        joints[f"{leg}_hip"] = hip

        # Knee goes downward
        # knee = hip + np.array([offset[0], -thigh_length, offset[2]])
        knee = hip + np.array([offset[0]*2, 0, offset[2]*2])
        joints[f"{leg}_knee"] = knee

        # Foot goes further down
        foot = knee + np.array([offset[0], -shin_length, offset[2]])
        joints[f"{leg}_foot"] = foot

        limbs.append(("torso", f"{leg}_hip"))
        limbs.append((f"{leg}_hip", f"{leg}_knee"))
        limbs.append((f"{leg}_knee", f"{leg}_foot"))

        for joint in [f"{leg}_hip", f"{leg}_knee"]:
            click_counts[joint] = 0

def draw_sphere(pos, radius=2, color=(1, 0, 0)):
    glPushMatrix()
    glTranslatef(*pos)
    glColor3fv(color)
    quad = gluNewQuadric()
    gluSphere(quad, radius, 20, 20)
    glPopMatrix()

def draw_limb(start, end, radius=0.25, colors=[1, 1, 1]):
    # Create a vector from start to end
    start = np.array(start)
    end = np.array(end)
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        return

    # Normalize direction
    direction /= length

    # Default cylinder points up the z-axis; we need to rotate it to match the direction
    up = np.array([0, 0, 1])
    axis = np.cross(up, direction)
    angle = math.degrees(math.acos(np.dot(up, direction)))
    
    # Draw the cylinder
    glPushMatrix()
    glTranslatef(*start)

    if np.linalg.norm(axis) > 1e-6:
        glRotatef(angle, *axis)

    glColor3f(*colors)
    quad = gluNewQuadric()
    gluCylinder(quad, radius, radius, length, 20, 1)
    glPopMatrix()


def render_ant():
    RADIUS_TORSO = 0.8
    RADIUS_LEGS = 0.3
    
    
    for joint, pos in joints.items():
        if "foot" in joint:
            draw_sphere(pos, RADIUS_LEGS, (1, 1, 1))
        elif "torso" in joint:
            count = click_counts.get(joint, 0)
            intensity = min(1.0, count * 0.25)
            color = (1.0, 1.0 - intensity, 1.0 - intensity)
            draw_sphere(pos, RADIUS_TORSO, color)
        else:
            count = click_counts.get(joint, 0)
            intensity = min(1.0, count * 0.25)
            color = (1.0, 1.0 - intensity, 1.0 - intensity)
            draw_sphere(pos, RADIUS_LEGS, color)

    for start, end in limbs:
        draw_limb(joints[start], joints[end], RADIUS_LEGS)

def get_mouse_ray(mx, my):
    x = (2.0 * mx) / WIDTH - 1.0
    y = 1.0 - (2.0 * my) / HEIGHT
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)

    near = gluUnProject(mx, HEIGHT - my, 0.0, modelview, projection, viewport)
    far = gluUnProject(mx, HEIGHT - my, 1.0, modelview, projection, viewport)

    ray_origin = np.array(near)
    ray_dir = np.array(far) - ray_origin
    ray_dir /= np.linalg.norm(ray_dir)
    return ray_origin, ray_dir

def ray_sphere_intersect(ray_origin, ray_dir, center, radius):
    oc = ray_origin - center
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - radius ** 2
    discriminant = b * b - 4 * c
    return discriminant >= 0

def handle_click(mx, my):
    ray_origin, ray_dir = get_mouse_ray(mx, my)
    for name, center in joints.items():
        if "foot" in name or "torso" in name:
            continue
        if ray_sphere_intersect(ray_origin, ray_dir, center, 0.4):
            click_counts[name] += 1
            break

def handle_mouse_motion(pos):
    global last_mouse_pos, rotation
    if mouse_down:
        dx = pos[0] - last_mouse_pos[0]
        dy = pos[1] - last_mouse_pos[1]
        rotation[0] += dx * 0.4
        rotation[1] += dy * 0.4
    last_mouse_pos = pos

def main():
    global mouse_down
    init()
    build_ant_structure()
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                mouse_down = True
                handle_click(*event.pos)
            elif event.type == MOUSEBUTTONUP and event.button == 1:
                mouse_down = False
            elif event.type == MOUSEMOTION:
                handle_mouse_motion(event.pos)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, -1, -10)
        glRotatef(rotation[1], 1, 0, 0)
        glRotatef(rotation[0], 0, 1, 0)
        render_ant()
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
