import pygame
pygame.init()
from pygame.locals import *

from random import randint
import math as m

# Planet Types:
# 0 - Solid Planet
# 1 - Gas Planet
# 2 - Small Star
# 3 - Big Star
# 4 - Black Hole

data = {
    "sun": {
        "col": (255, 255, 0),
        "rad": 50,
        "grav": 30,
        "pos": [800, 450],
        "vel": [0, 0],
        "type": 2
    },
    "earth": {
        "col": (100, 100, 255),
        "rad": 5,
        "grav": 1,
        "pos": [1100, 450],
        "vel": [0, -8],
        "type": 0
    },
    "venus": {
        "col": (255, 50, 50),
        "rad": 5,
        "grav": 0.9,
        "pos": [1000, 450],
        "vel": [0, -6],
        "type": 0
    },
    "mars": {
        "col": (255, 0, 0),
        "rad": 3,
        "grav": 0.4,
        "pos": [1200, 450],
        "vel": [0, -10],
        "type": 0
    },
    "jupiter": {
        "col": (255, 150, 150),
        "rad": 12,
        "grav": 2.4,
        "pos": [300, 450],
        "vel": [0, 12],
        "type": 1
    },
  
}

debris_data = {
    "debris1": {
        "col": (128, 0, 128),  # Color (RGB)
        "rad": 2,                # Radius
        "pos": [1000, 400],      # Initial position [x, y]
        "vel": [2, 0.5],         # Initial velocity [vx, vy] (pixels per frame)
    },
    "debris2": {
        "col": (255, 165, 0),
        "rad": 3,
        "pos": [800, 300],
        "vel": [-1, 1],
    },
    # Add more debris entries as needed
}
winSize = (1600, 900)
display = pygame.display.set_mode(winSize)
fps = 27
clock = pygame.time.Clock()

simSpeed = 0.01

def collision_outcome(type1, type2):
    if type1 == 0:
        if type2 == 0 or type2 == 1: return 0
    if type1 == 1:
        if type2 == 0 or type2 == 1: return 1
    if type1 == 2:
        if type2 == 0: return 2
        if type2 == 1 or type2 == 2: return 3
    if type1 == 3:
        if type2 == 0 or type2 == 1: return 3
        if type2 == 2 or type2 == 3: return 4

stars = [((randint(150, 200), randint(150, 200), randint(150, 200)), (randint(1, winSize[0]), randint(1, winSize[1])), randint(1, 2)) for _ in range(250)]
def draw_stars():
    for star in stars:
        pygame.draw.circle(display, star[0], star[1], star[2])

def draw_planets():
    for planet in data:
        pygame.draw.circle(display, data[planet]['col'], data[planet]['pos'], data[planet]['rad'])

#Jas IS HERE 
def draw_debris():
    for debris_name, debris_info in debris_data.items():
        pygame.draw.circle(display, debris_info['col'], debris_info['pos'], debris_info['rad'])
import csv

# Function to read debris data from CSV file
def read_debris_data_from_csv(filename):
    debris_data = {}
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            debris_name = row['Name']  # Assuming 'Name' is a column in your CSV file
            debris_data[debris_name] = {
                
                'col': (128, 0, 128),  #purple 
                'rad': float(row['Radius']),  # Radius
                'pos': [int(row['X']), int(row['Y'])],  # Initial position [x, y]
                'vel': [float(row['Vx']), float(row['Vy'])],  # Initial velocity [vx, vy]
            }
            # Add more characteristics as needed
    return debris_data

# Example usage
csv_filename = r'C:\Users\USER\Desktop\TechnicalGP\sim_debris.xlsx'
debris_data = read_debris_data_from_csv(csv_filename)

# Inside the main loop:
for debris_name, debris_info in debris_data.items():
    pygame.draw.circle(display, debris_info['col'], debris_info['pos'], debris_info['rad'])
    # Update position of debris based on velocity
    debris_info['pos'][0] += debris_info['vel'][0] * simSpeed
    debris_info['pos'][1] += debris_info['vel'][1] * simSpeed


def update_planets():
    toDelete = []
    for planet in data:
        for oPlanet in data:
            if oPlanet != planet:
                dx = data[oPlanet]['pos'][0] - data[planet]['pos'][0]
                dy = data[oPlanet]['pos'][1] - data[planet]['pos'][1]

                distance = m.sqrt(dx*dx + dy*dy)

                dx = dx / distance * data[oPlanet]['grav'] * simSpeed
                dy = dy / distance * data[oPlanet]['grav'] * simSpeed

                data[planet]['vel'][0] += dx
                data[planet]['vel'][1] += dy

                if distance <= data[oPlanet]['rad']:
                    toDelete.append(planet)
                    data[oPlanet]['grav'] += data[planet]['grav']
                    data[oPlanet]['rad'] += data[planet]['rad'] / 2
                    co = collision_outcome(data[oPlanet]['type'], data[planet]['type'])
                    if co == 0: data[oPlanet]['col'] = (150, 0, 0)
                    elif co == 1: data[oPlanet]['col'] = (255, 150, 150)
                    elif co == 2: data[oPlanet]['col'] = (255, 255, 0)
                    elif co == 3: data[oPlanet]['col'] = (255, 0, 0)
                    elif co == 4:
                        data[oPlanet]['col'] = (20, 20, 20)
                        data[oPlanet]['grav'] *= 2
                    data[oPlanet]['type'] = co

        data[planet]['pos'][0] += data[planet]['vel'][0]
        data[planet]['pos'][1] += data[planet]['vel'][1]

    for item in toDelete:
        del data[item]
def update_debris():
    for debris_info in debris_data.values():
        debris_info['pos'][0] += debris_info['vel'][0] * simSpeed
        debris_info['pos'][1] += debris_info['vel'][1] * simSpeed
        # Additional logic for handling collisions or interactions with other objects can be added here


run = True
while run:
    display.fill((0, 0, 0))

    draw_stars()
    draw_planets()
    update_planets()
    # Inside the main loop:
    draw_debris()
    update_debris()

    clock.tick(fps)
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == QUIT:
            run = False

pygame.quit()