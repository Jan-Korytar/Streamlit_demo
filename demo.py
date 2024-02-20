import streamlit as st
import pandas as pd
import numpy as np
from torch import tensor
import torch.nn.functional as F

# Note that most of the comments are chatGPT generated and pandas dataframes are used as an overhead from previous
# versions

with st.sidebar:

    # User can adjust the dimensions of the kernel using a slider
    st.subheader('Kernel to be used:')
    kernel_size = st.sidebar.slider("Dimensions of kernel_1", 1, 5, 3, )

    # Check if the kernel has been initialized or if the dimensions have changed
    if 'kernel' not in st.session_state or kernel_size != len(st.session_state['kernel'][0]):
        # Initialize a random kernel with the specified dimensions
        st.session_state['kernel'] = np.random.randint(0, 10, (kernel_size, kernel_size), dtype=np.uint8)

    # Retrieve the current kernel from the session state
    kernel = st.session_state['kernel']

    # Inform the user that the kernel is editable
    st.text('The kernel is editable by the user!')

    # Display the kernel in a data editor for user interaction
    dataframe = pd.DataFrame(kernel)
    data = st.data_editor(dataframe)

# Main title for the application
st.title("Simple kernel computation demonstration")

cols = st.columns(2)

with cols[0]:
    # User can enter the array width using a slider
    width = st.slider('Enter array width', 5, 30)

with cols[1]:
    # User can enter the array height using a slider
    height = st.slider('Enter array height', 5, 30)

st.write('Select the index where upper left corner of the kernel will be:')

with cols[0]:
    # User can enter the x index using a slider
    y_idx = st.slider('Enter x index ', 0, width - kernel_size // 2 - 2)
with cols[1]:
    # User can enter the y index using a slider
    x_idx = st.slider('Enter y index', 0, height - kernel_size // 2 - 2)
# Display a message indicating where the user can select the kernel's upper-left corner


# Highlight specific rows and columns in the DataFrame to visualize the selected area
highlighted_rows = [i for i in range(x_idx, x_idx + kernel_size)]
highlighted_cols = [f'{i}' for i in range(y_idx, y_idx + kernel_size)]


def highlight_cells(x, row_indices, col_indices):
    # Create a DataFrame of the same shape as x with all False values
    highlight = pd.DataFrame('', index=x.index, columns=x.columns)

    # Highlight the specified cells with a yellow background and black text
    for row_index in row_indices:
        for col_index in col_indices:
            highlight.at[row_index, col_index] = 'background-color: yellow; color: black'

    return highlight


# Specify the row and column indices to highlight
max = 5
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame(np.random.randint(0, max, (30, 30), dtype=np.uint8), columns=[f'{i}' for i in range(30)])
df = st.session_state['df']
df_selected = df.loc[range(height), [f'{i}' for i in range(width)]]
df_style = df_selected.style.apply(highlight_cells, row_indices=highlighted_rows, col_indices=highlighted_cols, axis=None)


# Display the DataFrame with highlighted cells
st.write(df_style)

chosen = df_selected.loc[highlighted_rows, highlighted_cols]
st.subheader('How is cross-correlation computed for one output cell')
cols = st.columns(3)
with cols[0]:
    st.write('selected input')
    st.write(chosen)
with cols[1]:
    st.write('selected output')
    st.write(data)
with cols[2]:
    st.write('elementwise multiplication')
    combined = chosen.astype(str).values + '*' + data.astype(str).values + '=' + (chosen.values * data.values).astype(str)
    st.write(combined)
    st.write(f'sum: {(chosen.values * data.astype(int).values).sum()}')


st.subheader('Result of the 2d convolution using PyTorch implementation')
st.write(tensor(df_selected.values).unsqueeze(0), tensor(data.values).unsqueeze(0))
conv_result = F.conv2d(tensor(df_selected.values)[None, None, ...], tensor(data.values)[None, None, ...], stride=1)
conv_df = pd.DataFrame(conv_result.squeeze().numpy())  # Assuming you want to convert the result to a NumPy array
st.write("Note that this is rather cross-correlation than convolution")
st.write(conv_df)

st.write("Used code")
with open('demo.py') as f:
    f = f.read()
    st.code(f, language='python')