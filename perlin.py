# Importing necessary libraries
import numpy as np  # for numerical operations
import matplotlib.pyplot as plt  # for plotting
from scipy.ndimage import gaussian_filter  # for applying gaussian filter
import noise  # for generating perlin noise

# Setting dimensions and scale for the noise maps
width = 1024
height = 1024
steps = 8  # Number of steps to quantize the noise map, affecting the smoothness


# Function to generate stepped perlin noise map
def generate_stepped_perlin_noise(map_width, map_height, scale, octaves, persistence, lacunarity, seed, num_steps):
    world = np.zeros((map_height, map_width))
    for i in range(map_height):
        for j in range(map_width):
            # Generate Perlin noise value for each pixel
            world[i][j] = noise.pnoise2(i / scale,
                                        j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=map_width,
                                        repeaty=map_height,
                                        base=seed)
    # Normalizing and stepping the world values
    world = (world - world.min()) / (world.max() - world.min())
    world = np.floor(world * num_steps) / num_steps
    return world


# Function to generate base noise map
def generate_base_noise(map_width, map_height, seed):
    np.random.seed(seed)
    return np.random.rand(map_width, map_height)


# Function to apply gaussian filter and step the continentalness map
def generate_stepped_continentalness_map(base_noise_map, sigma, num_steps):
    continentalness_result_map2 = gaussian_filter(base_noise_map, sigma=sigma)
    continentalness_result_map2 = (continentalness_result_map2 - continentalness_result_map2.min()) / (
            continentalness_result_map2.max() - continentalness_result_map2.min())
    continentalness_result_map2 = np.floor(continentalness_result_map2 * num_steps) / num_steps
    return continentalness_result_map2


# Generating base noise and maps for different geographical features
base_noise = generate_base_noise(width, height, seed=np.random.randint(0, 100))
continentalness_map = generate_stepped_continentalness_map(base_noise, sigma=16, num_steps=steps)

erosion_map = generate_stepped_perlin_noise(width, height, scale=50, octaves=6, persistence=0.5, lacunarity=2.0,
                                            seed=np.random.randint(0, 100), num_steps=steps)
peaks_valleys_map = generate_stepped_perlin_noise(width, height, scale=10, octaves=6, persistence=1, lacunarity=2.0,
                                                  seed=np.random.randint(0, 100), num_steps=steps)
temperature_map = generate_stepped_perlin_noise(width, height, scale=100, octaves=1, persistence=0.2, lacunarity=2.0,
                                                seed=np.random.randint(0, 100), num_steps=steps)
humidity_map = generate_stepped_perlin_noise(width, height, scale=50, octaves=4, persistence=0.7, lacunarity=2.0,
                                             seed=np.random.randint(0, 100), num_steps=steps)

# Setting up a figure with multiple plots for different noise maps
fig, axs = plt.subplots(2, 3, figsize=(10, 6))

# Plotting and formatting each map
axs[0, 0].imshow(continentalness_map, cmap='gray')
axs[0, 0].set_title('Continentalness')
axs[0, 0].axis('off')

axs[0, 1].imshow(erosion_map, cmap='gray')
axs[0, 1].set_title('Erosion')
axs[0, 1].axis('off')

axs[0, 2].imshow(peaks_valleys_map, cmap='gray')
axs[0, 2].set_title('Peaks & Valleys')
axs[0, 2].axis('off')

axs[1, 0].imshow(temperature_map, cmap='gray')
axs[1, 0].set_title('Temperature')
axs[1, 0].axis('off')

axs[1, 1].imshow(humidity_map, cmap='gray')
axs[1, 1].set_title('Humidity')
axs[1, 1].axis('off')

# Leaving the last subplot empty
axs[1, 2].axis('off')

# Adjusting layout and displaying the plot
plt.tight_layout()
plt.show()
