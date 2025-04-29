import pygame
import math

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MuJoCo Ant 2D Visualization")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BUTTON_COLOR = (100, 100, 255)

# Main torso position
torso_pos = (WIDTH // 2, HEIGHT // 2)

# Lengths of body parts
thigh_length = 60
shin_length = 60

# Define joints (dynamic)
joints = {}

# Define limbs (connections between joints)
limbs = []

# Click tracking
click_counts = {}

# Movie mode variables
movie_mode = False
movie_frame = 0

# Define button positions
restart_button = pygame.Rect(850, 50, 140, 40)
movie_button = pygame.Rect(850, 120, 140, 40)

# Joint/limb settings
joint_radius = 15  # Clickable joint size

def build_ant_structure(center):
    joints.clear()
    limbs.clear()
    click_counts.clear()

    cx, cy = center

    # Angles for each leg (starting direction from torso)
    angles = {
        "front_left": math.radians(45),
        "front_right": math.radians(135),
        "back_left": math.radians(-45),
        "back_right": math.radians(-135),
    }

    knee_offset_angle = math.radians(30)  # how much knee bends outward
    foot_offset_angle = math.radians(30)  # how much foot bends outward

    # Build legs
    for leg, angle in angles.items():
        # Hip
        hip_x = cx + 40 * math.cos(angle)
        hip_y = cy + 40 * math.sin(angle)

        # Knee (bent outward)
        if "left" in leg:
            knee_angle = angle + knee_offset_angle
            foot_angle = knee_angle + foot_offset_angle
        else:
            knee_angle = angle - knee_offset_angle
            foot_angle = knee_angle - foot_offset_angle

        knee_x = hip_x + thigh_length * math.cos(knee_angle)
        knee_y = hip_y + thigh_length * math.sin(knee_angle)

        # Foot (bend again from knee direction)
        foot_x = knee_x + shin_length * math.cos(foot_angle)
        foot_y = knee_y + shin_length * math.sin(foot_angle)

        # Register joints
        joints[f"{leg}_hip"] = (hip_x, hip_y)
        joints[f"{leg}_knee"] = (knee_x, knee_y)
        joints[f"{leg}_foot"] = (foot_x, foot_y)

        # Register limbs
        limbs.append(("torso", f"{leg}_hip"))
        limbs.append((f"{leg}_hip", f"{leg}_knee"))
        limbs.append((f"{leg}_knee", f"{leg}_foot"))

    # Initialize click counts for joints that are clickable (hips and knees)
    for joint in joints:
        if "foot" not in joint:
            click_counts[joint] = 0

def draw_ant():
    # Draw torso
    pygame.draw.circle(screen, BLACK, torso_pos, 30)

    # Draw limbs (thicker, same width as joints)
    for start, end in limbs:
        start_pos = torso_pos if start == "torso" else joints[start]
        end_pos = joints[end]
        pygame.draw.line(screen, BLACK, start_pos, end_pos, joint_radius)

    # Draw joints
    for name, pos in joints.items():
        if "foot" in name:
            color = (0, 0, 0)  # Feet not clickable
            radius = 10
        else:
            intensity = min(255, click_counts[name] * 40)
            color = (255, 255 - intensity, 255 - intensity)
            radius = joint_radius

        # First draw black outline slightly bigger
        pygame.draw.circle(screen, BLACK, (int(pos[0]), int(pos[1])), radius + 2)
        # Then draw actual colored circle inside
        pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), radius)

def draw_buttons():
    pygame.draw.rect(screen, BUTTON_COLOR, restart_button)
    pygame.draw.rect(screen, BUTTON_COLOR, movie_button)

    restart_text = font.render('Restart', True, WHITE)
    movie_text = font.render('Run Movie', True, WHITE)

    screen.blit(restart_text, (restart_button.x + 30, restart_button.y + 5))
    screen.blit(movie_text, (movie_button.x + 8, movie_button.y + 5))

def reset_clicks():
    for joint in click_counts:
        click_counts[joint] = 0

def run_movie():
    global movie_mode, movie_frame
    movie_mode = True
    movie_frame = 0

# Initialize the ant
build_ant_structure(torso_pos)

running = True
movie_mode = False
while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()

            # Check joint clicks
            for name, (x, y) in joints.items():
                if "foot" not in name:
                    if (x - pos[0])**2 + (y - pos[1])**2 < joint_radius**2:
                        click_counts[name] += 1
                        break
            
            # Check button clicks
            if restart_button.collidepoint(pos):
                reset_clicks()
            if movie_button.collidepoint(pos):
                movie_mode = True

    if movie_mode:
        print(click_counts)
        movie_mode = False


    draw_ant()
    draw_buttons()

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
