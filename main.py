import streamlit as st
import pandas as pd
import numpy as np

with open("project_areas.txt") as f:
    project_areas = [pa.strip() for pa in f.readlines()]

df = pd.read_pickle('df.pkl')
points = np.load('points.npy')

st.write("# FYP Project Area Selector")

st.write("""
This application lists previous year's final year, HDip, and MSc projects 
ordered by project areas. 
It may or may not help :smile: you narrow down your FYP selection by 
seeing what projects are similar to yours based your chosen project areas.
""")

st.info("""
Technical details (for those of you taking Bernard's Data Mining 1 module): The similarity score used in the ordering 
is based on the Euclidean distance.  Each of last year's projects are represented a vertex in the $n$-dimensional unit cuboid 
($n$ is number of project areas). Projects with equal scores are ordered alphabetically based on title.
""",icon="ℹ️")


st.write("""## Project Areas
For each of the project areas, select how important/relevant you expect it to be in your FYP. 
The ranking of previous year FYP is done automatically based on your selection.
""")


N_COLUMNS = 3
n = len(project_areas)
columns = st.columns(N_COLUMNS)
project_areas_sliders = [None] * n


help_text = lambda pa: f"Select degree of importance this project area ({pa}) has to your project."
for c in range(N_COLUMNS):
    with columns[c]:
        for k in range(c,n,N_COLUMNS):
            pa = project_areas[k]
            slider = st.slider(pa, min_value=0, max_value=100, step=5, key=pa, format="%d%%")
            project_areas_sliders[k] = slider

# compute this project vertex
point = np.array([float(pas)/100 for pas in project_areas_sliders])

distances = np.linalg.norm(points - point, axis=1)

df['Distance'] = distances

df_tmp = df.sort_values(['Distance','AcademicTitle']).reset_index(drop=False)

st.write("## Last Year Projects")

num_display = st.slider("Select number of projects to display:",
                        min_value=1, max_value=df.shape[0], step=1, value=5)

st.write("Click on project title to see project details and project URL ...")

for idx, row in df_tmp.iterrows():
    if idx==num_display:break
    with st.expander(f"**{row.AcademicTitle}**"):
        # st.write("#### Commercial Title")
        # st.write(row.CommercialTitle)
        # st.write("#### Academic Title")
        # st.write(row.AcademicTitle)
        st.write("#### Summary")
        st.write(row.Summary)
        st.write("#### Technologies")
        st.write(row.Technologies)
        st.write("#### Project URL")
        st.write(row.ProjectURL)
        st.write("#### Project Areas")
        st.write(row.ProjectAreas)

        # st.write("#### Similarity Score")
        # st.write(f"{row.Distance:.4f}")