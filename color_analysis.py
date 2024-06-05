import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import colorsys
import webcolors
import pandas as pd

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def closest_color(requested_color):
    min_colors = {}
    color_names = []
    for color in requested_color:
        min_dist = float('inf')
        closest_name = ""
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - color[0]) ** 2
            gd = (g_c - color[1]) ** 2
            bd = (b_c - color[2]) ** 2
            dist = rd + gd + bd
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        color_names.append(closest_name)
    return color_names

def get_dominant_colors(image, k=5):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.array(image)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)

    colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    label_counts = Counter(labels)
    total_count = sum(label_counts.values())

    dominant_colors = []
    for color, count in label_counts.items():
        percent = (count / total_count) * 100
        color_rgb = colors[color].astype(int)
        color_hex = rgb_to_hex(tuple(color_rgb))
        color_name = closest_color([color_rgb])[0]
        dominant_colors.append({
            'color': color_hex,
            'name': color_name,
            'percentage': percent
        })

    dominant_colors.sort(key=lambda x: x['percentage'], reverse=True)  # Sort by percentage
    return dominant_colors

def get_matching_colors(color_rgb):
    color_hex = rgb_to_hex(tuple(color_rgb))
    r, g, b = color_rgb
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)

    # Generate analogous colors
    analogous_colors = []
    for shift in [-0.2, 0, 0.2]:
        h_shift = (h + shift) % 1.0
        analogous_rgb = colorsys.hls_to_rgb(h_shift, l, s)
        analogous_hex = rgb_to_hex((int(analogous_rgb[0] * 255), int(analogous_rgb[1] * 255), int(analogous_rgb[2] * 255)))
        analogous_colors.append(analogous_hex)

    # Generate complementary color
    complementary_h = (h + 0.5) % 1.0
    complementary_rgb = colorsys.hls_to_rgb(complementary_h, l, s)
    complementary_hex = rgb_to_hex((int(complementary_rgb[0] * 255), int(complementary_rgb[1] * 255), int(complementary_rgb[2] * 255)))

    return analogous_colors + [complementary_hex]

def ignore_background(image, k=5):
    # Get the dominant colors of the entire image
    dominant_colors = get_dominant_colors(image, k=k)
    # Sort the dominant colors by percentage
    dominant_colors.sort(key=lambda x: x['percentage'], reverse=True)
    # Assuming the background color is the most dominant color
    background_color = webcolors.hex_to_rgb(dominant_colors[0]['color'])
    # Ignore the background color and get dominant colors again
    dominant_colors = get_dominant_colors(image, k=k)
    filtered_colors = [color for color in dominant_colors if np.linalg.norm(np.array(webcolors.hex_to_rgb(color['color'])) - np.array(background_color)) > 50]
    return filtered_colors

st.title("Color Analysis Tool")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Analyze'):
        dominant_colors = ignore_background(image)
        if dominant_colors:
            primary_color = dominant_colors[0]
            st.write("**Primary Color:**")
            st.write(f"- Color: {primary_color['color']}")
            st.write(f"- Name: {primary_color['name']}")
            st.write(f"- Percentage: {primary_color['percentage']}%")

            secondary_colors = dominant_colors[1:]
            st.write("**Secondary Colors:**")
            secondary_color_data = [(color['color'], color['name'], f"{color['percentage']}%") for color in secondary_colors]
            secondary_color_df = pd.DataFrame(secondary_color_data, columns=["Color", "Name", "Percentage"])
            st.dataframe(secondary_color_df)

            primary_color_rgb = np.array(webcolors.hex_to_rgb(primary_color['color']))
            matching_colors = get_matching_colors(primary_color_rgb)
            matching_names = closest_color([np.array(webcolors.hex_to_rgb(match)) for match in matching_colors if np.any(primary_color_rgb != np.array(webcolors.hex_to_rgb(match)))])
            st.write("**Matching Colors:**")
            matching_color_data = []
            for match, name in zip(matching_colors, matching_names):
                matching_color_data.append((match, name))
            matching_color_df = pd.DataFrame(matching_color_data, columns=["Matching Color", "Matching Color Name"])
            st.dataframe(matching_color_df)
        else:
            st.write("Unable to find dominant colors.")
