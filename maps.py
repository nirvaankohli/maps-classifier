import streamlit as st
from streamlit.components.v1 import html

st.set_page_config(
    page_title="Satellite Map Embed",
    layout="wide",
)

st.title("🛰️ Satellite Imagery")



center = [0, 0]
zoom = 13

html_string = f"""

<link

  rel="stylesheet"
  href="https://unpkg.com/leaflet/dist/leaflet.css"

/>

<div id="map" style="width:100%; height:600px;"></div>

<script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>

<script>

  var map = L.map('map').setView([{center[0]}, {center[1]}], {zoom});

  L.tileLayer(

    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
     attribution: 'Tiles &copy; Esri &mdash; Source: Esri, USGS, USDA'
    
    }}

  ).addTo(map);

  
</script>

"""

html(
  
  html_string, 

  height=650
  
  )
