import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time
import json
import os
from datetime import datetime
import argparse

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"
DEFAULT_FIGSIZE = (12, 16)
DEFAULT_DPI = 300

DEFAULT_THEME = {
    "name": "Feature-Based Shading",
    "bg": "#FFFFFF",
    "text": "#000000",
    "gradient_color": "#FFFFFF",
    "water": "#C0C0C0",
    "road_motorway": "#0A0A0A",
    "road_primary": "#1A1A1A",
    "road_secondary": "#2A2A2A",
    "road_tertiary": "#3A3A3A",
    "road_residential": "#4A4A4A",
    "road_default": "#3A3A3A"
}

def load_fonts():
    """
    Load Roboto fonts from the fonts directory.
    Returns dict with font paths for different weights.
    """
    fonts = {
        'bold': os.path.join(FONTS_DIR, 'Roboto-Bold.ttf'),
        'regular': os.path.join(FONTS_DIR, 'Roboto-Regular.ttf'),
        'light': os.path.join(FONTS_DIR, 'Roboto-Light.ttf')
    }
    
    # Verify fonts exist
    for weight, path in fonts.items():
        if not os.path.exists(path):
            print(f"⚠ Font not found: {path}")
            return None
    
    return fonts

FONTS = load_fonts()

def slugify_label(label):
    slug = label.lower().strip()
    for char in [' ', ',', '/']:
        slug = slug.replace(char, '_')
    return slug

def generate_output_filename(location_label, theme_name):
    """
    Generate unique output filename with city, theme, and datetime.
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_slug = slugify_label(location_label)
    filename = f"{label_slug}_{theme_name}_{timestamp}.png"
    return os.path.join(POSTERS_DIR, filename)

def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []
    
    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith('.json'):
            theme_name = file[:-5]  # Remove .json extension
            themes.append(theme_name)
    return themes

def blend_color(color, target, amount):
    base = np.array(mcolors.to_rgb(color))
    target_rgb = np.array(mcolors.to_rgb(target))
    blended = base * (1 - amount) + target_rgb * amount
    return mcolors.to_hex(blended, keep_alpha=False)

def apply_theme_defaults(theme):
    base_defaults = {
        "name": DEFAULT_THEME["name"],
        "bg": DEFAULT_THEME["bg"],
        "text": DEFAULT_THEME["text"],
        "gradient_color": DEFAULT_THEME["gradient_color"]
    }
    for key, value in base_defaults.items():
        theme.setdefault(key, value)
    return theme

def load_theme(theme_name="feature_based"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")
    
    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default feature_based theme.")
        return DEFAULT_THEME.copy()
    
    with open(theme_file, 'r') as f:
        theme = json.load(f)
        print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
        if 'description' in theme:
            print(f"  {theme['description']}")
        return apply_theme_defaults(theme)

# Load theme (can be changed via command line or input)
THEME = None  # Will be loaded later

def create_gradient_fade(ax, color, location='bottom', zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))
    
    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]
    
    if location == 'bottom':
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    
    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end
    
    ax.imshow(gradient, extent=[xlim[0], xlim[1], y_bottom, y_top], 
              aspect='auto', cmap=custom_cmap, zorder=zorder, origin='lower')

def normalize_highway(highway):
    if isinstance(highway, list):
        return highway[0] if highway else 'unclassified'
    return highway

def theme_has(key):
    return key in THEME and THEME[key] not in (None, '')

def theme_color(key):
    return THEME[key] if theme_has(key) else None

def get_road_color(highway):
    highway = normalize_highway(highway)
    if highway in ['motorway', 'motorway_link']:
        return theme_color('road_motorway')
    if highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
        return theme_color('road_primary')
    if highway in ['secondary', 'secondary_link']:
        return theme_color('road_secondary')
    if highway in ['tertiary', 'tertiary_link']:
        return theme_color('road_tertiary')
    if highway in ['residential', 'living_street', 'unclassified']:
        return theme_color('road_residential')
    return theme_color('road_default')

def get_road_width(highway):
    highway = normalize_highway(highway)
    if highway in ['motorway', 'motorway_link']:
        return 1.2 if theme_has('road_motorway') else 0
    if highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
        return 1.0 if theme_has('road_primary') else 0
    if highway in ['secondary', 'secondary_link']:
        return 0.8 if theme_has('road_secondary') else 0
    if highway in ['tertiary', 'tertiary_link']:
        return 0.6 if theme_has('road_tertiary') else 0
    return 0.4 if theme_has('road_residential') or theme_has('road_default') else 0

def get_edge_colors_by_type(G):
    """
    Assigns colors to edges based on road type hierarchy.
    Returns a list of colors corresponding to each edge in the graph.
    """
    colors = []
    for _, _, data in G.edges(data=True):
        color = get_road_color(data.get('highway', 'unclassified'))
        colors.append(color if color else THEME['bg'])
    return colors

def get_edge_widths_by_type(G):
    """
    Assigns line widths to edges based on road type.
    Major roads get thicker lines.
    """
    return [get_road_width(data.get('highway', 'unclassified')) for _, _, data in G.edges(data=True)]

def get_coordinates(city, country):
    """
    Fetches coordinates for a given city and country using geopy.
    Includes rate limiting to be respectful to the geocoding service.
    """
    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster")
    
    # Add a small delay to respect Nominatim's usage policy
    time.sleep(1)
    
    location = geolocator.geocode(f"{city}, {country}")
    
    if location:
        print(f"✓ Found: {location.address}")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        return (location.latitude, location.longitude)
    else:
        raise ValueError(f"Could not find coordinates for {city}, {country}")

def parse_center(center_value):
    if not center_value:
        return None
    parts = center_value.split(',')
    if len(parts) != 2:
        raise ValueError("Center must be in 'lat,lon' format.")
    lat = float(parts[0].strip())
    lon = float(parts[1].strip())
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Center coordinates are out of range.")
    return (lat, lon)

def resolve_figsize(width_in, height_in):
    if width_in <= 0 or height_in <= 0:
        raise ValueError("Width and height must be positive values.")
    return (width_in, height_in)

def filter_geometry_types(gdf, allowed_types):
    if gdf is None or gdf.empty:
        return gdf
    return gdf[gdf.geometry.geom_type.isin(allowed_types)]

def create_poster(city, country, point, dist, output_file, figsize, dpi, show_text, show_gradient):
    print(f"\nGenerating map for {city}, {country}...")
    
    # Progress bar for data fetching
    with tqdm(total=7, desc="Fetching map data", unit="step", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # 1. Fetch Street Network
        pbar.set_description("Downloading street network")
        G = ox.graph_from_point(
            point,
            dist=dist,
            dist_type='bbox',
            network_type='all',
            truncate_by_edge=True,
            retain_all=True
        )
        pbar.update(1)
        time.sleep(0.5)  # Rate limit between requests
        
        # 2. Fetch Water Features
        pbar.set_description("Downloading water features")
        water = None
        if theme_has('water'):
            try:
                water = ox.features_from_point(point, tags={'natural': 'water', 'waterway': 'riverbank'}, dist=dist)
            except:
                water = None
        pbar.update(1)
        time.sleep(0.3)
        
        # 3. Fetch Railways
        pbar.set_description("Downloading railways")
        railways = None
        if theme_has('railway'):
            try:
                railways = ox.features_from_point(point, tags={'railway': ['rail', 'light_rail', 'tram', 'subway', 'narrow_gauge', 'monorail']}, dist=dist)
            except:
                railways = None
        pbar.update(1)
        time.sleep(0.2)

        # 4. Fetch Waterways
        pbar.set_description("Downloading waterways")
        waterways = None
        if theme_has('waterway'):
            try:
                waterways = ox.features_from_point(point, tags={'waterway': ['river', 'stream', 'canal', 'drain', 'ditch']}, dist=dist)
            except:
                waterways = None
        pbar.update(1)
        time.sleep(0.2)

        # 5. Fetch Buildings + Contours
        pbar.set_description("Downloading buildings and contours")
        buildings = None
        contours = None
        if theme_has('buildings'):
            try:
                buildings = ox.features_from_point(point, tags={'building': True}, dist=dist)
            except:
                buildings = None
        if theme_has('contours'):
            try:
                contours = ox.features_from_point(point, tags={'natural': 'contour'}, dist=dist)
            except:
                contours = None
        pbar.update(1)
        time.sleep(0.2)

        # 6. Fetch Subways
        pbar.set_description("Downloading subways")
        subways = None
        if theme_has('subway'):
            try:
                subways = ox.features_from_point(point, tags={'railway': 'subway'}, dist=dist)
            except:
                subways = None
        pbar.update(1)
        time.sleep(0.2)

        # 7. Fetch Airports + Runways
        pbar.set_description("Downloading airports and runways")
        airports = None
        runways = None
        if theme_has('airport'):
            try:
                airports = ox.features_from_point(point, tags={'aeroway': ['aerodrome', 'terminal', 'apron']}, dist=dist)
            except:
                airports = None
        if theme_has('runway'):
            try:
                runways = ox.features_from_point(point, tags={'aeroway': ['runway', 'taxiway']}, dist=dist)
            except:
                runways = None
        pbar.update(1)
    
    print("✓ All data downloaded successfully!")
    
    # 2. Setup Plot
    print("Rendering map...")
    fig, ax = plt.subplots(figsize=figsize, facecolor=THEME['bg'])
    ax.set_facecolor(THEME['bg'])
    ax.set_position([0, 0, 1, 1])
    
    # 3. Plot Layers
    # Layer 1: Polygons
    water_polygons = filter_geometry_types(water, ['Polygon', 'MultiPolygon'])
    airport_polygons = filter_geometry_types(airports, ['Polygon', 'MultiPolygon'])
    building_polygons = filter_geometry_types(buildings, ['Polygon', 'MultiPolygon'])

    waterways_lines = filter_geometry_types(waterways, ['LineString', 'MultiLineString'])
    runways_lines = filter_geometry_types(runways, ['LineString', 'MultiLineString'])
    subways_lines = filter_geometry_types(subways, ['LineString', 'MultiLineString'])
    railways_lines = filter_geometry_types(railways, ['LineString', 'MultiLineString'])
    contours_lines = filter_geometry_types(contours, ['LineString', 'MultiLineString'])

    if theme_has('water') and water_polygons is not None and not water_polygons.empty:
        water_polygons.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=1)
    if theme_has('airport') and airport_polygons is not None and not airport_polygons.empty:
        airport_polygons.plot(ax=ax, facecolor=THEME['airport'], edgecolor='none', zorder=1.4)
    if theme_has('buildings') and building_polygons is not None and not building_polygons.empty:
        building_polygons.plot(ax=ax, facecolor=THEME['buildings'], edgecolor='none', zorder=2.3)
    if theme_has('contours') and contours_lines is not None and not contours_lines.empty:
        contours_lines.plot(ax=ax, color=THEME['contours'], linewidth=0.4, alpha=0.6, zorder=2.6)
    if theme_has('waterway') and waterways_lines is not None and not waterways_lines.empty:
        waterways_lines.plot(ax=ax, color=THEME['waterway'], linewidth=0.6, zorder=2.8)
    if theme_has('runway') and runways_lines is not None and not runways_lines.empty:
        runways_lines.plot(ax=ax, color=THEME['runway'], linewidth=2, zorder=2.85)
    if theme_has('subway') and subways_lines is not None and not subways_lines.empty:
        subways_lines.plot(ax=ax, color=THEME['subway'], linewidth=0.5, zorder=2.87)
    if theme_has('railway') and railways_lines is not None and not railways_lines.empty:
        railways_lines.plot(ax=ax, color=THEME['railway'], linewidth=1, zorder=2.9)
    
    # Layer 2: Roads with hierarchy coloring
    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(G)
    edge_widths = get_edge_widths_by_type(G)
    if any(width > 0 for width in edge_widths):
        ox.plot_graph(
            G, ax=ax, bgcolor=THEME['bg'],
            node_size=0,
            edge_color=edge_colors,
            edge_linewidth=edge_widths,
            show=False, close=False
        )

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True, fill_edge_geometry=True)
    if edges is not None and not edges.empty:
        def filter_edges_by_tag(edge_gdf, tag):
            if tag not in edge_gdf.columns:
                return edge_gdf.iloc[0:0]
            values = edge_gdf[tag].fillna('').astype(str).str.lower()
            return edge_gdf[~values.isin(['', 'no', '0', 'false'])]

        bridge_edges = filter_edges_by_tag(edges, 'bridge')
        tunnel_edges = filter_edges_by_tag(edges, 'tunnel')

        if theme_has('tunnel') and not tunnel_edges.empty:
            tunnel_widths = [get_road_width(h) for h in tunnel_edges.get('highway', [])]
            tunnel_edges.plot(
                ax=ax,
                color=THEME['tunnel'],
                linewidth=tunnel_widths,
                linestyle='--',
                alpha=0.7,
                zorder=3.4
            )
        if theme_has('bridge') and not bridge_edges.empty:
            bridge_widths = [get_road_width(h) + 0.2 for h in bridge_edges.get('highway', [])]
            bridge_edges.plot(
                ax=ax,
                color=THEME['bridge'],
                linewidth=bridge_widths,
                zorder=3.5
            )

    
    # Layer 3: Gradients (Top and Bottom)
    if show_gradient:
        create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
        create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)
    
    # 4. Typography using Roboto font
    if show_text:
        if FONTS:
            font_main = FontProperties(fname=FONTS['bold'], size=60)
            font_sub = FontProperties(fname=FONTS['light'], size=22)
            font_coords = FontProperties(fname=FONTS['regular'], size=14)
        else:
            # Fallback to system fonts
            font_main = FontProperties(family='monospace', weight='bold', size=60)
            font_sub = FontProperties(family='monospace', weight='normal', size=22)
            font_coords = FontProperties(family='monospace', size=14)
        
        spaced_city = "  ".join(list(city.upper()))

        # --- BOTTOM TEXT ---
        ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
                color=THEME['text'], ha='center', fontproperties=font_main, zorder=11)
        
        ax.text(0.5, 0.10, country.upper(), transform=ax.transAxes,
                color=THEME['text'], ha='center', fontproperties=font_sub, zorder=11)
        
        lat, lon = point
        coords = f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {lon:.4f}° E"
        if lon < 0:
            coords = coords.replace("E", "W")
        
        ax.text(0.5, 0.07, coords, transform=ax.transAxes,
                color=THEME['text'], alpha=0.7, ha='center', fontproperties=font_coords, zorder=11)
        
        ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes, 
                color=THEME['text'], linewidth=1, zorder=11)

        # --- ATTRIBUTION (bottom right) ---
        if FONTS:
            font_attr = FontProperties(fname=FONTS['light'], size=8)
        else:
            font_attr = FontProperties(family='monospace', size=8)
        
        ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
                color=THEME['text'], alpha=0.5, ha='right', va='bottom', 
                fontproperties=font_attr, zorder=11)

    # 5. Save
    print(f"Saving to {output_file}...")
    plt.savefig(output_file, dpi=dpi, facecolor=THEME['bg'])
    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")

def print_examples():
    """Print usage examples."""
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]
  python create_map_poster.py --center <lat,lon> [options]

Examples:
  # Iconic grid patterns
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 12000           # Manhattan grid
  python create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000   # Eixample district grid
  
  # Waterfront & canals
  python create_map_poster.py -c "Venice" -C "Italy" -t blueprint -d 4000       # Canal network
  python create_map_poster.py -c "Amsterdam" -C "Netherlands" -t ocean -d 6000  # Concentric canals
  python create_map_poster.py -c "Dubai" -C "UAE" -t midnight_blue -d 15000     # Palm & coastline
  
  # Radial patterns
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000   # Haussmann boulevards
  python create_map_poster.py -c "Moscow" -C "Russia" -t noir -d 12000          # Ring roads
  
  # Organic old cities
  python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000    # Dense organic streets
  python create_map_poster.py -c "Marrakech" -C "Morocco" -t terracotta -d 5000 # Medina maze
  python create_map_poster.py -c "Rome" -C "Italy" -t warm_beige -d 8000        # Ancient street layout
  
  # Coastal cities
  python create_map_poster.py -c "San Francisco" -C "USA" -t sunset -d 10000    # Peninsula grid
  python create_map_poster.py -c "Sydney" -C "Australia" -t ocean -d 12000      # Harbor city
  python create_map_poster.py -c "Mumbai" -C "India" -t contrast_zones -d 18000 # Coastal peninsula
  
  # River cities
  python create_map_poster.py -c "London" -C "UK" -t noir -d 15000              # Thames curves
  python create_map_poster.py -c "Budapest" -C "Hungary" -t copper_patina -d 8000  # Danube split
  
  # List themes
  python create_map_poster.py --list-themes
  
  # Center on coordinates (no geocoding)
  python create_map_poster.py --center "40.7128,-74.0060" --distance 12000 --theme noir

Options:
  --city, -c        City name (required)
  --country, -C     Country name (required)
  --center          Map center as "lat,lon" (optional)
  --theme, -t       Theme name (default: feature_based)
  --distance, -d    Map radius in meters (default: 29000)
  --width           Poster width in inches (default: 12)
  --height          Poster height in inches (default: 16)
  --dpi             Output DPI (default: 300)
  --no-text         Render map without labels or attribution
  --no-gradient     Render map without top/bottom gradient fades
  --list-themes     List all available themes

Distance guide:
  4000-6000m   Small/dense cities (Venice, Amsterdam old center)
  8000-12000m  Medium cities, focused downtown (Paris, Barcelona)
  15000-20000m Large metros, full city view (Tokyo, Mumbai)

Available themes can be found in the 'themes/' directory.
Generated posters are saved to 'posters/' directory.
""")

def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return
    
    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, 'r') as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except:
            display_name = theme_name
            description = ''
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster.py --city "New York" --country "USA"
  python create_map_poster.py --city Tokyo --country Japan --theme midnight_blue
  python create_map_poster.py --city Paris --country France --theme noir --distance 15000
  python create_map_poster.py --center "40.7128,-74.0060" --distance 12000 --theme noir
  python create_map_poster.py --list-themes
        """
    )
    
    parser.add_argument('--city', '-c', type=str, help='City name')
    parser.add_argument('--country', '-C', type=str, help='Country name')
    parser.add_argument('--center', type=str, help='Map center as "lat,lon" (optional)')
    parser.add_argument('--theme', '-t', type=str, default='feature_based', help='Theme name (default: feature_based)')
    parser.add_argument('--distance', '-d', type=int, default=29000, help='Map radius in meters (default: 29000)')
    parser.add_argument('--width', type=float, default=DEFAULT_FIGSIZE[0], help='Poster width in inches (default: 12)')
    parser.add_argument('--height', type=float, default=DEFAULT_FIGSIZE[1], help='Poster height in inches (default: 16)')
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI, help='Output DPI (default: 300)')
    parser.add_argument('--no-text', action='store_true', help='Render map without labels or attribution')
    parser.add_argument('--no-gradient', action='store_true', help='Render map without top/bottom gradient fades')
    parser.add_argument('--list-themes', action='store_true', help='List all available themes')
    
    args = parser.parse_args()
    
    # If no arguments provided, show examples
    if len(os.sys.argv) == 1:
        print_examples()
        os.sys.exit(0)
    
    # List themes if requested
    if args.list_themes:
        list_themes()
        os.sys.exit(0)
    
    # Validate required arguments
    if not args.center and (not args.city or not args.country):
        print("Error: --city and --country are required unless --center is provided.\n")
        print_examples()
        os.sys.exit(1)
    
    # Validate theme exists
    available_themes = get_available_themes()
    if args.theme not in available_themes:
        print(f"Error: Theme '{args.theme}' not found.")
        print(f"Available themes: {', '.join(available_themes)}")
        os.sys.exit(1)
    
    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)
    
    # Load theme
    THEME = load_theme(args.theme)
    
    # Get coordinates and generate poster
    try:
        center_point = parse_center(args.center)
        if center_point:
            coords = center_point
            display_city = args.city or "Custom Location"
            display_country = args.country or ""
            location_label = args.city or f"center_{coords[0]:.4f}_{coords[1]:.4f}"
        else:
            display_city = args.city
            display_country = args.country
            coords = get_coordinates(args.city, args.country)
            location_label = args.city

        figsize = resolve_figsize(args.width, args.height)
        output_file = generate_output_filename(location_label, args.theme)
        create_poster(display_city, display_country, coords, args.distance, output_file, figsize, args.dpi, not args.no_text, not args.no_gradient)
        
        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        os.sys.exit(1)
