import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

def plot_graph(coordinates, line_names, xlabel, ylabel, title, show_values, cutin_jumpin_data, max_x_axis, max_y_axis):

    fig, ax = plt.subplots(figsize=(12, 7))

    # Safely extract cut-in and jump-in data
    if len(cutin_jumpin_data) > 0:
        cut_in_x = cutin_jumpin_data['X'][0]
        cut_in_tol = cutin_jumpin_data['X Tol'][0]
        cut_in_brackets = cutin_jumpin_data['X Brackets'][0]
        show_cut_in = cutin_jumpin_data['Show X Val'][0]

        jump_in_y = cutin_jumpin_data['Y'][0]
        jump_in_tol = cutin_jumpin_data['Y Tol'][0]
        jump_in_brackets = cutin_jumpin_data['Y Brackets'][0]
        show_jump_in = cutin_jumpin_data['Show Y Val'][0]
    else:
        cut_in_x = cut_in_tol = 0
        cut_in_brackets = False
        show_cut_in = False
        jump_in_y = jump_in_tol = 0
        jump_in_brackets = False
        show_jump_in = False

    # Set axis limits from input
    ax.set_xlim(0, max_x_axis)
    ax.set_ylim(0, max_y_axis)

    # Plot solid black lines for cut-in and jump-in extending to axes
    if show_cut_in:
        # Vertical solid line from (cut_in_x, 0) up to (cut_in_x, jump_in_y)
        ax.plot([cut_in_x, cut_in_x], [0, jump_in_y], color='black', linewidth=1.5)
        # Label for cut-in placed below x-axis at cut_in_x
        label_cut_in = f"Cut-in: {cut_in_x:.1f}"
        if cut_in_tol > 0:
            label_cut_in += f"±{cut_in_tol:.1f}"
        if cut_in_brackets:
            label_cut_in = f"({label_cut_in})"
        ax.text(cut_in_x, -0.08 * max_y_axis, label_cut_in, ha='center', va='top', fontsize=8, color='black')

    if show_jump_in:
        # Horizontal solid line from (0, jump_in_y) to (cut_in_x, jump_in_y)
        ax.plot([0, cut_in_x], [jump_in_y, jump_in_y], color='black', linewidth=1.5)
        # Label for jump-in placed left of y-axis at jump_in_y
        label_jump_in = f"Jump-in: {jump_in_y:.1f}"
        if jump_in_tol > 0:
            label_jump_in += f"±{jump_in_tol:.1f}"
        if jump_in_brackets:
            label_jump_in = f"({label_jump_in})"
        ax.text(-0.07 * max_x_axis, jump_in_y, label_jump_in, ha='right', va='center', fontsize=8, color='black')

    # Plot the main lines with labels
    for i, line_data in enumerate(coordinates):
        x_coords = line_data['X']
        y_coords = line_data['Y']

        ax.plot(x_coords, y_coords, color='black', linewidth=1)

        if i == 0 and len(x_coords) > 0:
            # vertical line from x start to y start on first line
            ax.plot([x_coords[0], x_coords[0]], [0, y_coords[0]], color='black', linewidth=1)

        # Calculate angle for label rotation
        if len(x_coords) > 1:
            dx = x_coords[-1] - x_coords[0]
            dy = y_coords[-1] - y_coords[0]
            angle = np.degrees(np.arctan2(dy, dx))
        else:
            angle = 0
            dx = dy = 0

        mid_x = np.mean(x_coords) if len(x_coords) > 0 else 0
        mid_y = np.mean(y_coords) if len(y_coords) > 0 else 0

        length = np.hypot(dx, dy) if len(x_coords) > 1 else 1
        if length != 0:
            normal_x = -dy / length
            normal_y = dx / length
            offset = 0.03 * max_y_axis
            offset_x = normal_x * offset
            offset_y = normal_y * offset + 0.01 * max_y_axis * i  # vertical offset for multiple labels
        else:
            offset_x = offset_y = 0

        ax.text(mid_x + offset_x, mid_y + offset_y, line_names[i],
                ha='center', va='center', fontsize=12, color='black',
                rotation=angle - 2, rotation_mode='anchor')

    # Draw dashed grid lines for each point
    for line_data in coordinates:
        for x, y in zip(line_data['X'], line_data['Y']):
            ax.plot([x, x], [0, y], 'k--', linewidth=0.8)
            ax.plot([0, x], [y, y], 'k--', linewidth=0.8)

    # Format labels for axis values with tolerance and brackets
    def format_val_label(value, tol, brackets):
        label = f"{value:.1f}"
        if tol > 0:
            label += f"±{tol:.1f}"
        if brackets:
            label = f"({label})"
        return label

    # Adjust subplot to have enough margin for labels
    fig.subplots_adjust(left=0.20, right=0.78, top=0.78, bottom=0.22)

    if show_values:
        # Prepare and plot Y-axis labels on left side (exclude jump_in_y)
        y_points = []
        for line_data in coordinates:
            for y, tol, br, show_y in zip(line_data['Y'], line_data['Y Tol'], line_data['Y Brackets'], line_data['Show Y Val']):
                if show_y and y != jump_in_y:
                    y_points.append((y, format_val_label(y, tol, br)))

        # No need to add jump_in_y here as it's labeled alongside line

        y_points = sorted(y_points, key=lambda x: x[0])

        y_min, y_max = ax.get_ylim()
        y_norm = [((y - y_min) / (y_max - y_min), label) for y, label in y_points]

        # Avoid overlapping labels by spacing them vertically
        y_stack = []
        gap = 0.03
        for y_pos, label in y_norm:
            if y_stack and abs(y_stack[-1][0] - y_pos) < gap:
                y_pos_new = y_stack[-1][0] + gap
                y_stack.append((y_pos_new, label))
            else:
                y_stack.append((y_pos, label))

        x_left = 0.15  # figure coords for left side labels
        for y_pos, label in y_stack:
            fig.text(x_left, 0.18 + y_pos * 0.6, label,
                     ha='right', va='center', fontsize=9, color='black')

        # Prepare and plot X-axis labels on bottom (exclude cut_in_x)
        x_points = []
        for line_data in coordinates:
            for x, tol, br, show_x in zip(line_data['X'], line_data['X Tol'], line_data['X Brackets'], line_data['Show X Val']):
                if show_x and x != cut_in_x:
                    x_points.append((x, format_val_label(x, tol, br)))

        # No need to add cut_in_x here as it's labeled alongside line

        x_points = sorted(x_points, key=lambda x: x[0])

        x_min, x_max = ax.get_xlim()

        stacked = []
        gap = 0.035
        for idx, (x_val, label) in enumerate(x_points):
            x_norm = (x_val - x_min) / (x_max - x_min)
            if stacked and abs(stacked[-1][0] - x_norm) < gap:
                y_off = 0.15 + ((-1) ** idx) * gap
                stacked.append((x_norm, y_off, label))
            else:
                stacked.append((x_norm, 0.15, label))

        for x_norm, y_pos, label in stacked:
            fig.text(0.22 + x_norm * 0.56, y_pos, label,
                     ha='center', va='top', fontsize=9, color='black')

    ax.set_xlabel(xlabel, labelpad=40, fontsize=15)
    ax.set_ylabel(ylabel, labelpad=130, fontsize=15)
    ax.set_title(title, pad=30, fontsize=10)

    # Set nice ticks
    x_tick_max = np.ceil(max_x_axis / 10) * 10
    y_tick_max = np.ceil(max_y_axis / 10) * 10
    ax.set_xticks(np.arange(0, x_tick_max + 1, 10))
    ax.set_yticks(np.arange(0, y_tick_max + 1, 10))

    ax.grid(True)

    st.pyplot(fig)

    # Save figure to BytesIO buffer for download or reuse
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    plt.close(fig)
    return buf


# Streamlit interface
st.title('Graph Plotter')

xlabel = st.text_input('Enter X-axis label:', 'INPUT FORCE --> Kg')
ylabel = st.text_input('Enter Y-axis label:', 'OUTPUT PRESSURE Bar --> bar')
title = st.text_area('Enter graph title:', 'PERFORMANCE CHARACTERISTICS\nTOLERANCE BAND TO BE FINALISED AFTER DATA\nGENERATION ON LARGE NO. OF SAMPLES')

# New cut-in and jump-in input as table (only one row expected)
cutin_jumpin_df = pd.DataFrame({
    'X': [0.0],
    'Y': [0.0],
    'X Tol': [0.0],
    'Y Tol': [0.0],
    'X Brackets': [False],
    'Y Brackets': [False],
    'Show X Val': [True],
    'Show Y Val': [True]
})
st.write("Enter Cut-in (X axis) and Jump-in (Y axis) points (including tolerances):")
cutin_jumpin_df = st.data_editor(cutin_jumpin_df, num_rows='dynamic', key='cutin_jumpin')

num_lines = st.number_input('How many lines are there in the graph?', min_value=1, step=1)

line_names = []
coordinates = []

for i in range(num_lines):
    line_name = st.text_input(f'Enter the name for line {i+1}:', key=f'line_name_{i}')
    line_names.append(line_name)

    num_coords = st.number_input(f'How many coordinates are there for line {i+1}?', min_value=1, step=1, key=f'num_coords_{i}')

    df = pd.DataFrame({
        'X': [0.0] * num_coords,
        'Y': [0.0] * num_coords,
        'X Tol': [0.0] * num_coords,
        'Y Tol': [0.0] * num_coords,
        'X Brackets': [False] * num_coords,
        'Y Brackets': [False] * num_coords,
        'Show X Val': [True] * num_coords,
        'Show Y Val': [True] * num_coords,
    })

    st.write(f'Enter coordinates, tolerances, brackets and toggles for line {i+1}:')
    edited_df = st.data_editor(df, num_rows='dynamic', key=f'df_{i}')

    line_data = {
        'X': edited_df['X'].tolist(),
        'Y': edited_df['Y'].tolist(),
        'X Tol': edited_df['X Tol'].tolist(),
        'Y Tol': edited_df['Y Tol'].tolist(),
        'X Brackets': edited_df['X Brackets'].tolist(),
        'Y Brackets': edited_df['Y Brackets'].tolist(),
        'Show X Val': edited_df['Show X Val'].tolist(),
        'Show Y Val': edited_df['Show Y Val'].tolist(),
    }
    coordinates.append(line_data)

# Defaults for max axis values based on input data (or 50 if no data)
default_max_x = max([max(line['X']) if line['X'] else 0 for line in coordinates], default=0)
default_max_y = max([max(line['Y']) if line['Y'] else 0 for line in coordinates], default=0)

max_x_axis = st.number_input('Enter maximum X-axis length:', min_value=10.0, value=max(default_max_x, 50.0), step=10.0)
max_y_axis = st.number_input('Enter maximum Y-axis length:', min_value=10.0, value=max(default_max_y, 50.0), step=10.0)

show_values = st.checkbox('Show coordinate values on graph', value=True)

if st.button('Generate Graph'):
    buf = plot_graph(coordinates, line_names, xlabel, ylabel, title, show_values, cutin_jumpin_df, max_x_axis, max_y_axis)
    st.download_button(
        label='Download Graph',
        data=buf,
        file_name='Graph.png',
        mime='image/png'
    )
