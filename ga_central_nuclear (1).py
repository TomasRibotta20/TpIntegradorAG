"""
Algoritmo Genético para determinar la ubicación óptima de una
Central Nuclear Modular Pequeña (SMR) a lo largo del corredor
Río Paraná -> Mar del Plata.
 
Criterios de fitness:
  1. Costo de construcción
       a) Tendido eléctrico: costo estimado de construir la línea de
          alta tensión hasta la Estación Transformadora más cercana.
       b) Envío de materiales: costo estimado de transporte marítimo
          de componentes hasta el sitio, en función de la distancia
          al puerto más cercano.
  2. Densidad poblacional por anillos alrededor del sitio:
       - Dentro de las 2 millas: densidad debe ser MUY BAJA (restricción dura)
       - Entre 2 y 10 millas: densidad MEDIA tolerada
       - Más allá de las 10 millas (hasta 50 millas y más): sin restricción
 
NOTA IMPORTANTE:
  Los valores de población, costos por km de tendido y de transporte,
  y las coordenadas de puertos/aeropuertos son APROXIMACIONES para que
  el modelo funcione de punta a punta. Para un uso real, reemplazar:
    - CIUDADES (poblacion) por datos censales oficiales (INDEC).
    - El modelo de densidad "gravitatorio" por datos grillados reales
      (ej. WorldPop / GPW) vía rasterio.
    - COSTO_TENDIDO_USD_KM, COSTO_TRANSPORTE_MARITIMO_USD_KM por
      cotizaciones reales de EPC / logística.
    - ESTACIONES_TRANSFORMADORAS por el listado real de estaciones
      transformadoras (ET) de la red de alta tensión (no subestaciones
      de distribución).
 
Salidas:
  - mapa_optimo.html   -> mapa interactivo (Leaflet/Folium) con el punto óptimo
  - convergencia_ga.png -> evolución del fitness (mejor y promedio) por generación
"""
 
import math
import random
import json
import os
from datetime import datetime
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
 
random.seed(42)
np.random.seed(42)
 
 
# DATOS DE ENTRADA (REEMPLAZAR CON TUS DATOS REALES)
 
BOUNDARY_POINTS = [
  (-35.146836,-57.3536387 ),	
  (-35.0167268,-57.5254052 ),
  (-35.0003861,-57.5809272 ),
  (-34.9692072,-57.6278766 ),
  (-34.933399,-57.6892455 )	,
  (-34.9279807,-57.7210887 ),
  (-34.9032936,-57.7709109 ),
  (-34.8309227,-57.8738219 ),
  (-34.8335299,-57.9338176 ),
  (-34.8250227,-57.9604034 ),
  (-34.7813095,-58.014524 ),
  (-34.7763746,-58.0564952 ),
  (-34.7516959,-58.11589 ),
  (-34.7478877,-58.1710791 ),
  (-34.7362504,-58.1976867 ),
  (-34.7158706,-58.2151523 ),
  (-34.6680319,-58.3001247 ),
  (-34.6472756,-58.3323971 ),
  (-34.5791088,-58.3784881 ),
  (-34.5318891,-58.4636322 ),
  (-34.4478454,-58.5189071 ),
  (-34.3318726,-58.4454823 ),
  (-34.2408164,-58.7747288 ),
  (-34.1854536,-58.9051914 ),
  (-34.1374432,-58.9865589 ),
  (-33.805457,-59.3583775 ),
  (-32.9425889,-60.6216154 ),
  (-32.5105103,-60.775424 ),
  (-31.7284854,-60.636435 ),
  (-31.1473454,-59.9325347 ),
  (-30.0171303,-59.6029448 ),
]
 
 
# Ciudades: (nombre, lat, lon, poblacion_aprox_hab)
# Poblaciones aproximadas (ciudad/aglomerado), a modo de orden de magnitud.
CIUDADES = [
    ("Corrientes",     -27.47, -58.83,  360_000),
    ("Santa Fe",       -31.63, -60.70,  415_000),
    ("Rosario",        -32.95, -60.65, 1_000_000),
    ("San Nicolás",    -33.34, -60.21,  145_000),
    ("Zárate",         -34.10, -59.03,  130_000),
    ("Buenos Aires",   -34.61, -58.38, 3_000_000),
    ("La Plata",       -34.92, -57.95,  740_000),
    ("Mar del Plata",  -38.00, -57.55,  620_000),
]
 
# Estaciones Transformadoras (ET) de la red de alta tensión
# (para costo de tendido eléctrico). NO son subestaciones de
# distribución: son los nodos de la red de transporte de energía
# (AT/EAT) a los que se conectaría la línea de la central.
 
ESTACIONES_TRANSFORMADORAS = [
    ("ET Resistencia", -27.45, -59.00),
    ("ET Santo Tomé", -31.67, -60.78),
    ("ET Rosario Oeste", -32.90, -60.75),
    ("ET Ezeiza", -34.85, -58.55),
    ("ET Necochea", -38.55, -58.74),
]
 
# Puertos (para costo de transporte de materiales pesados)
 
PUERTOS = [
    ("Puerto Buenos Aires", -34.6023, -58.3644),
    ("Puerto Rosario", -32.9200, -60.6500),
    ("Puerto La Plata", -34.8500, -57.9000),
    ("Puerto Zárate", -34.0900, -59.0000),
    ("Puerto Santa Fe", -31.6500, -60.6800),
    ("Puerto Quequén / Necochea", -38.5500, -58.7000),
    ("Puerto San Nicolás", -33.3300, -60.2000),
]
 
# Aeropuertos (solo referencia visual en el mapa; ya NO se usan para
# el cálculo de costo, que ahora es únicamente marítimo)
 
AEROPUERTOS = [
    ("Aeropuerto Rosario - Islas Malvinas", -32.9036, -60.7853),
    ("Aeropuerto Ezeiza - Ministro Pistarini", -34.8222, -58.5358),
    ("Aeroparque Jorge Newbery", -34.5592, -58.4156),
    ("Aeropuerto Santa Fe", -31.7104, -60.7810),
    ("Aeropuerto Mar del Plata", -37.9342, -57.5733),
]
 
 
# FUNCIONES GEOGRÁFICAS
 
 
def haversine(p1, p2):
    """Distancia en km entre dos puntos (lat, lon)."""
    R = 6371.0
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
 
 
def destino(point, bearing_deg, distancia_km):
    """Punto de destino a partir de un punto, un rumbo (grados) y una distancia (km)."""
    R = 6371.0
    lat1, lon1 = map(math.radians, point)
    brng = math.radians(bearing_deg)
    d_r = distancia_km / R
    lat2 = math.asin(math.sin(lat1) * math.cos(d_r) + math.cos(lat1) * math.sin(d_r) * math.cos(brng))
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d_r) * math.cos(lat1),
        math.cos(d_r) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lat2), math.degrees(lon2))
 
 
def interpolate_point(t, boundary_points):
  # Interpolación lineal sobre la polilínea de la línea frontera.
    t = min(max(t, 0.0), 1.0)
    n = len(boundary_points)
    idx_float = t * (n - 1)
    i = int(idx_float)
    frac = idx_float - i
    if i >= n - 1:
        return boundary_points[-1]
    p1, p2 = boundary_points[i], boundary_points[i + 1]
    lat = p1[0] + frac * (p2[0] - p1[0])
    lon = p1[1] + frac * (p2[1] - p1[1])
    return (lat, lon)
 
 
 
# DENSIDAD POBLACIONAL — GHS-POP (Global Human Settlement Population Grid)
#
# Fuente: GHS-POP R2023A, Comisión Europea / JRC.
# Producto:    GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0
# Proyección:  World Mollweide (ESRI:54009)
# Resolución:  1 km (cada celda = población estimada del km² de esa celda)
# Escenario:   E2030 (proyección de población a 2030)
# Descarga:    https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php
#
# En vez de un único .tif global, se arma un mosaico en memoria a partir
# de todos los tiles .tif que estén en la carpeta GHS_POP_DIR (ver abajo).
# Para el corredor de este proyecto hacen falta los tiles: R13_C12,
# R13_C13, R14_C12, R14_C13 (nomenclatura
# GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0_R##_C##.tif).
# NoData = -200 (celdas sin datos / fuera de tierra firme).
 
import glob
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge as rio_merge
from pyproj import Transformer
 
GHS_POP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghs_pop_tiles")
GHS_POP_NODATA = -200.0
 
# Definición EXPLÍCITA de la proyección Mollweide estándar usada por
# GHS-POP (World Mollweide, ESRI:54009), en vez de confiar en el CRS que
# viene embebido en cada .tif. Algunas versiones de PROJ/GDAL no
# resuelven bien el código "ESRI:54009" y terminan usando parámetros
# ligeramente distintos (radio/elipsoide), lo que produce un corrimiento
# horizontal sistemático que crece con la distancia al meridiano de
# Greenwich -justo lo que se ve en el corredor Paraná -> Mar del Plata,
# a ~55-60° de longitud oeste.
GHS_POP_PROJ4 = "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs"
 
MILLA_KM = 1.60934
RADIO_ZONA_BAJA_KM = 2 * MILLA_KM     # ~3.22 km
RADIO_ZONA_MEDIA_KM = 10 * MILLA_KM   # ~16.09 km
RADIO_ZONA_LIBRE_KM = 50 * MILLA_KM   # ~80.47 km (límite conceptual, sin restricción)
 
UMBRAL_DENSIDAD_BAJA = 40.0    # hab/km² máx. tolerado dentro de las 2 millas
UMBRAL_DENSIDAD_MEDIA = 400.0  # hab/km² máx. tolerado entre 2 y 10 millas
 
_ghs_pop_dataset = None
_ghs_pop_memfile = None  # hay que mantener viva la referencia al MemoryFile
_transformer_wgs84_a_mollweide = None
 
 
def _obtener_raster_ghs_pop():
    """
    Arma (una sola vez) un mosaico en memoria a partir de todos los .tif
    en GHS_POP_DIR y lo abre como un dataset de rasterio. El resto del
    código (densidad_en_puntos, recorte para el mapa, etc.) sigue
    trabajando exactamente igual que con un único .tif, porque el
    dataset resultante expone la misma interfaz (.sample, .read,
    .crs, .transform, .window_transform, ...).
    """
    global _ghs_pop_dataset, _ghs_pop_memfile, _transformer_wgs84_a_mollweide
    if _ghs_pop_dataset is None:
        archivos = sorted(glob.glob(os.path.join(GHS_POP_DIR, "*.tif")))
        if not archivos:
            raise FileNotFoundError(
                f"No se encontraron tiles GHS-POP (.tif) en '{GHS_POP_DIR}'.\n"
                "Colocá ahí los tiles que cubren el corredor de estudio "
                "(por ejemplo R13_C12, R13_C13, R14_C12, R14_C13; "
                "nomenclatura GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0_R##_C##.tif), "
                "descargados de https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php"
            )
 
        datasets_origen = [rasterio.open(f) for f in archivos]
        mosaico, transform = rio_merge(datasets_origen, nodata=GHS_POP_NODATA)
        crs_original = datasets_origen[0].crs
        for ds in datasets_origen:
            ds.close()
 
        # Se fuerza la definición estándar de Mollweide (ver GHS_POP_PROJ4)
        # en vez de crs_original, para evitar el corrimiento sistemático
        # que puede introducir una interpretación ambigua de "ESRI:54009".
        crs = GHS_POP_PROJ4
        print(f"[GHS-POP] CRS embebido en los tiles: {crs_original}")
        print(f"[GHS-POP] CRS forzado para la reproyección: {crs}")
 
        perfil = dict(
            driver="GTiff",
            height=mosaico.shape[1],
            width=mosaico.shape[2],
            count=1,
            dtype=mosaico.dtype,
            crs=crs,
            transform=transform,
            nodata=GHS_POP_NODATA,
        )
        _ghs_pop_memfile = MemoryFile()
        with _ghs_pop_memfile.open(**perfil) as dst:
            dst.write(mosaico)
        _ghs_pop_dataset = _ghs_pop_memfile.open()
 
        # always_xy=True para trabajar en orden (lon, lat) -> (x, y)
        _transformer_wgs84_a_mollweide = Transformer.from_crs(
            "EPSG:4326", _ghs_pop_dataset.crs, always_xy=True
        )
 
        print(f"[GHS-POP] Mosaico armado a partir de {len(archivos)} tile(s): "
              f"{[os.path.basename(a) for a in archivos]}")
        print(f"[GHS-POP] Forma del mosaico: {mosaico.shape[1]}x{mosaico.shape[2]} px")
 
    return _ghs_pop_dataset, _transformer_wgs84_a_mollweide
 
 
def densidad_en_puntos(points):
    """
    Devuelve la densidad poblacional (hab/km², aprox.) de una lista de
    puntos (lat, lon), leyendo el valor de celda del raster GHS-POP
    (cada celda de 1km x 1km en proyección equiárea Mollweide ~ 1 km²,
    por lo que la población de la celda ≈ densidad en hab/km²).
    """
    dataset, transformer = _obtener_raster_ghs_pop()
    xs, ys = transformer.transform([lon for _, lon in points], [lat for lat, _ in points])
    valores = [v[0] for v in dataset.sample(zip(xs, ys))]
    return [0.0 if (v is None or v <= GHS_POP_NODATA) else max(float(v), 0.0) for v in valores]
 
 
def densidad_en_punto(point):
    return densidad_en_puntos([point])[0]
 
 
def densidad_maxima_en_anillo(point, r_min_km, r_max_km, n_radios=5, n_angulos=16):
    """Estima la densidad máxima (peor caso) dentro de un anillo [r_min, r_max] km.
    n_radios/n_angulos suben respecto de la versión original (3x8=24 puntos)
    a 5x16=80 puntos por defecto, para no saltear picos de densidad
    localizados dado que la resolución del raster es de ~1 km/celda."""
    if r_min_km <= 0:
        radios = np.linspace(max(r_max_km * 0.15, 0.1), r_max_km, n_radios)
    else:
        radios = np.linspace(r_min_km, r_max_km, n_radios)
    puntos = [point]  # incluye el propio centro del sitio
    for r in radios:
        for ang in np.linspace(0, 360, n_angulos, endpoint=False):
            puntos.append(destino(point, ang, r))
    return max(densidad_en_puntos(puntos))


def evaluar_restriccion_poblacional(point):
    """
    Devuelve (fitness_poblacional [0,1], cumple_restriccion_dura, detalle)
    según las 3 zonas:
      - 0-2 millas:  densidad muy baja -> RESTRICCIÓN DURA (ver abajo)
      - 2-10 millas: densidad media tolerada -> restricción blanda
      - >10 millas:  sin restricción

    La zona 0-2 millas se muestrea con más resolución (n_radios/n_angulos
    más altos) porque es la que define si el sitio es viable o no.
    """
    dens_zona_baja = densidad_maxima_en_anillo(
        point, 0.0, RADIO_ZONA_BAJA_KM, n_radios=6, n_angulos=20
    )
    dens_zona_media = densidad_maxima_en_anillo(
        point, RADIO_ZONA_BAJA_KM, RADIO_ZONA_MEDIA_KM, n_radios=5, n_angulos=16
    )

    cumple_restriccion_dura = dens_zona_baja <= UMBRAL_DENSIDAD_BAJA

    if cumple_restriccion_dura:
        f_zona_baja = 1.0
    else:
        # Decae MUCHO más rápido que antes (factor 8 en el exponente, vs 1
        # en la versión original) para que ningún ahorro de costo pueda
        # compensar el haber superado el umbral de densidad muy baja.
        exceso_relativo = (dens_zona_baja - UMBRAL_DENSIDAD_BAJA) / UMBRAL_DENSIDAD_BAJA
        f_zona_baja = math.exp(-8.0 * exceso_relativo)

    if dens_zona_media <= UMBRAL_DENSIDAD_MEDIA:
        f_zona_media = 1.0
    else:
        f_zona_media = math.exp(-(dens_zona_media - UMBRAL_DENSIDAD_MEDIA) / UMBRAL_DENSIDAD_MEDIA)

    # La zona de 0-2 millas pesa más porque es la restricción dura de seguridad.
    fitness = 0.65 * f_zona_baja + 0.35 * f_zona_media
    detalle = dict(
        dens_zona_baja=dens_zona_baja,
        dens_zona_media=dens_zona_media,
        f_zona_baja=f_zona_baja,
        f_zona_media=f_zona_media,
        cumple_restriccion_dura=cumple_restriccion_dura,
    )
    return fitness, cumple_restriccion_dura, detalle
 
 
 
# COSTO DE CONSTRUCCIÓN
#
# a) Tendido eléctrico hasta la Estación Transformadora más cercana.
# b) Envío de materiales: costo de transporte marítimo únicamente,
#    en función de la distancia al puerto más cercano.
 
COSTO_TENDIDO_USD_KM = 300_000               # USD/km de línea de alta tensión (placeholder)
COSTO_TRANSPORTE_MARITIMO_USD_KM = 5_000     # USD/km desde el puerto (placeholder)
 
 
def costo_tendido_electrico(point):
    d_red = min(haversine(point, (e[1], e[2])) for e in ESTACIONES_TRANSFORMADORAS)
    return d_red * COSTO_TENDIDO_USD_KM
 
 
def costo_envio_materiales(point):
    """Costo de envío de materiales considerando únicamente transporte
    marítimo (distancia al puerto más cercano)."""
    d_puerto = min(haversine(point, (p[1], p[2])) for p in PUERTOS)
    return d_puerto * COSTO_TRANSPORTE_MARITIMO_USD_KM
 
 
def costo_construccion_bruto(point):
    """Costo total estimado en USD (tendido + envío de materiales)."""
    return costo_tendido_electrico(point) + costo_envio_materiales(point)
 
 
def normalizar(valores):
    vmin, vmax = min(valores), max(valores)
    rango = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1e-9
    return [(v - vmin) / rango for v in valores]
 
 
 
# ALGORITMO GENÉTICO
 
POP_SIZE = 60
N_GENERACIONES = 100
PROB_CRUZA = 0.75
PROB_MUTACION = 0.25
SIGMA_MUTACION_INICIAL = 0.12
TAM_TORNEO = 2
 
PESOS_FITNESS = dict(costo=0.5, poblacional=0.5)

# Penalización multiplicativa aplicada al fitness total cuando el sitio
# viola la restricción DURA de densidad (0-2 millas). Al ser multiplicativa
# y muy chica, ningún ahorro de costo alcanza para compensarla.
PENALIZACION_RESTRICCION_DURA = 0.02
 
 
def crear_individuo():
    return {"t": random.random()}
 
 
def evaluar_poblacion(poblacion):
    """Devuelve fitness normalizado [0,1] para toda la población.

    La restricción de densidad 0-2 millas es DURA: si un individuo la
    viola, su fitness total se multiplica por PENALIZACION_RESTRICCION_DURA
    (muy chico), de modo que ningún ahorro de costo pueda compensar haber
    superado el umbral. No se pone en 0 exacto para conservar algo de
    gradiente y que el AG pueda seguir "empujando" al individuo fuera de
    la zona prohibida en vez de quedar con fitness plano.
    """
    puntos = [interpolate_point(ind["t"], BOUNDARY_POINTS) for ind in poblacion]

    costos_brutos = [costo_construccion_bruto(p) for p in puntos]
    costos_norm = normalizar(costos_brutos)  # 0 = más barato

    fitness_total = []
    detalle = []
    for p, c_norm in zip(puntos, costos_norm):
        f_costo = 1 - c_norm
        f_poblacional, cumple_restriccion_dura, det_pob = evaluar_restriccion_poblacional(p)
        f = (PESOS_FITNESS["costo"] * f_costo
             + PESOS_FITNESS["poblacional"] * f_poblacional)
        if not cumple_restriccion_dura:
            f *= PENALIZACION_RESTRICCION_DURA
        fitness_total.append(f)
        detalle.append(dict(
            punto=p,
            f_costo=f_costo,
            f_poblacional=f_poblacional,
            costo_construccion_usd=costo_construccion_bruto(p),
            **det_pob,
        ))
    return fitness_total, detalle
 
def seleccion_torneo(poblacion, fitness, k=TAM_TORNEO):
    participantes = random.sample(list(zip(poblacion, fitness)), k)
    participantes.sort(key=lambda x: x[1], reverse=True)
    return participantes[0][0]
 
 
def cruza(p1, p2):
    if random.random() > PROB_CRUZA:
        return dict(p1), dict(p2)
    alpha = random.random()
    t1 = alpha * p1["t"] + (1 - alpha) * p2["t"]
    t2 = alpha * p2["t"] + (1 - alpha) * p1["t"]
    return {"t": min(max(t1, 0), 1)}, {"t": min(max(t2, 0), 1)}
 
 
def mutacion(ind, sigma):
    if random.random() < PROB_MUTACION:
        nuevo_t = ind["t"] + random.gauss(0, sigma)
        ind["t"] = min(max(nuevo_t, 0.0), 1.0)
    return ind
 
 
def correr_ga():
    poblacion = [crear_individuo() for _ in range(POP_SIZE)]
    historia_mejor = []
    historia_promedio = []
    historia_peor = []
    historia_std = []
    mejor_global = None
    mejor_fitness_global = -1
 
    for gen in range(N_GENERACIONES):
        fitness, detalle = evaluar_poblacion(poblacion)
 
        idx_mejor = int(np.argmax(fitness))
        if fitness[idx_mejor] > mejor_fitness_global:
            mejor_fitness_global = fitness[idx_mejor]
            mejor_global = dict(poblacion[idx_mejor])
            mejor_global["detalle"] = detalle[idx_mejor]
 
        historia_mejor.append(fitness[idx_mejor])
        historia_promedio.append(float(np.mean(fitness)))
        historia_peor.append(float(np.min(fitness)))
        historia_std.append(float(np.std(fitness)))
 
        # sigma de mutación decreciente (exploración -> explotación)
        sigma = 0.25
        orden = sorted(zip(poblacion, fitness), key=lambda x: x[1], reverse=True)
        nueva_poblacion = []
 
        while len(nueva_poblacion) < POP_SIZE:
            padre1 = seleccion_torneo(poblacion, fitness)
            padre2 = seleccion_torneo(poblacion, fitness)
            hijo1, hijo2 = cruza(padre1, padre2)
            hijo1 = mutacion(hijo1, sigma)
            hijo2 = mutacion(hijo2, sigma)
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < POP_SIZE:
                nueva_poblacion.append(hijo2)
 
        poblacion = nueva_poblacion
 
    return mejor_global, historia_mejor, historia_promedio, historia_peor, historia_std, poblacion
 
 
 
# GRÁFICO DE CONVERGENCIA 
 
 
def graficar_convergencia(historia_mejor, historia_promedio, historia_peor, historia_std,
                            path="convergencia_ga.png"):
    generaciones = np.arange(1, len(historia_mejor) + 1)
    promedio = np.array(historia_promedio)
    std = np.array(historia_std)
 
    fig, ax = plt.subplots(figsize=(10, 6))
 
    # Banda de +/- 1 desvío estándar alrededor del promedio
    ax.fill_between(generaciones, np.clip(promedio - std, 0, 1), np.clip(promedio + std, 0, 1),
                     color="#e65100", alpha=0.15, label="Promedio ± 1 desvío estándar")
 
    ax.plot(generaciones, historia_mejor, label="Mejor fitness", linewidth=2.2, color="#1b5e20")
    ax.plot(generaciones, historia_promedio, label="Fitness promedio", linewidth=2,
            linestyle="--", color="#e65100")
    ax.plot(generaciones, historia_peor, label="Peor fitness", linewidth=1.6,
            linestyle=":", color="#b71c1c")
 
    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness (0 - 1)")
    ax.set_title("Convergencia del Algoritmo Genético\nUbicación óptima - Central Nuclear Modular")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
 
    # Cuadro de texto con los valores finales (última generación)
    texto = (f"Última generación:\n"
              f"Mejor:    {historia_mejor[-1]:.3f}\n"
              f"Promedio: {historia_promedio[-1]:.3f}\n"
              f"Peor:     {historia_peor[-1]:.3f}\n"
              f"Desvío σ: {historia_std[-1]:.3f}")
    ax.text(0.015, 0.02, texto, transform=ax.transAxes, fontsize=9.5,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))
 
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
 
 
 
# MAPA HTML CON EL PUNTO ÓPTIMO
#
# La densidad poblacional ya NO se dibuja como una capa (ImageOverlay)
# generada a mano a partir del mosaico GHS-POP local: ese enfoque exigía
# reproyectar Mollweide -> WGS84 nosotros mismos y cualquier pequeño
# desvío en esa reproyección (o en el bounds pasado a folium) terminaba
# viéndose como un corrimiento horizontal de la capa respecto de las
# ciudades/costas reales.
#
# En su lugar, el propio MAPA BASE ya trae la densidad poblacional
# incorporada: se sirve como teselas (tiles) XYZ pre-generadas y
# perfectamente georreferenciadas en Web Mercator (EPSG:3857) por NASA
# GIBS, a partir de Gridded Population of the World v4 (GPWv4, CIESIN /
# SEDAC). Al ser tiles ya proyectadas por el propio servidor (mismo
# esquema que usan OpenStreetMap/Google), no hay reproyección casera de
# por medio y no se puede desalinear.
#
# Catálogo / documentación: https://nasa-gibs.github.io/gibs-api-docs/
GIBS_GPW_LAYER = "GPW_Population_Density_2020"  # años disponibles: 2000/2005/2010/2015/2020
GIBS_GPW_TILEMATRIXSET = "GoogleMapsCompatible_Level7"  # nivel de zoom nativo máx.: 7
GIBS_GPW_TILES_URL = (
    f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
    f"{GIBS_GPW_LAYER}/default/{GIBS_GPW_TILEMATRIXSET}/{{z}}/{{y}}/{{x}}.png"
)
GIBS_GPW_ATTR = (
    "Population density: NASA GIBS / CIESIN-SEDAC GPWv4 (2020) &mdash; "
    "<a href='https://earthdata.nasa.gov/gibs' target='_blank'>NASA EOSDIS GIBS</a>"
)
GIBS_GPW_LEGEND_URL = f"https://gibs.earthdata.nasa.gov/legends/{GIBS_GPW_LAYER}_H.svg"
 
 
def _leyenda_densidad_html():
    """Leyenda oficial de GIBS para la capa de densidad poblacional (SVG servido por NASA)."""
    return f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 8px 10px; border: 1px solid #888;
                border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);">
      <img src="{GIBS_GPW_LEGEND_URL}" alt="Densidad poblacional (hab/km²)" style="display:block; max-width: 320px;">
    </div>
    """
 
 
def generar_mapa(mejor, poblacion_final, fitness_total, path="mapa_optimo.html"):
    punto_optimo = mejor["detalle"]["punto"]
 
    # Sin tiles por defecto: el mapa base ES la capa de densidad poblacional
    # (NASA GIBS / GPWv4), no una capa aparte superpuesta a un mapa de calles.
    m = folium.Map(location=punto_optimo, zoom_start=6, tiles=None)
 
    folium.TileLayer(
        tiles=GIBS_GPW_TILES_URL,
        attr=GIBS_GPW_ATTR,
        name="Densidad poblacional (NASA GIBS - GPW 2020)",
        overlay=False,
        control=True,
        show=True,
        min_zoom=1,
        max_zoom=18,
        max_native_zoom=7,  # las tiles de GIBS solo existen hasta z=7; más zoom = mismo pixel escalado
    ).add_to(m)
 
    # Mapa base alternativo (calles/topónimos) por si se quiere ubicar
    # referencias de calles; se conmuta desde el control de capas, no se
    # superpone a la densidad.
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Mapa base (OpenStreetMap, sin densidad)",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)
 
    m.get_root().html.add_child(folium.Element(_leyenda_densidad_html()))
 
    # Línea frontera (corredor de ubicaciones disponibles)
    folium.PolyLine(
        BOUNDARY_POINTS, color="blue", weight=3, opacity=0.6,
        tooltip="Corredor disponible (Río Paraná -> Mar del Plata)"
    ).add_to(m)
 
    # Ciudades
    for nombre, lat, lon, poblacion in CIUDADES:
        folium.CircleMarker(
            location=(lat, lon), radius=5, color="#6a1b9a", fill=True,
            fill_color="#6a1b9a", fill_opacity=0.8,
            popup=f"Ciudad: {nombre} (~{poblacion:,} hab)".replace(",", "."),
        ).add_to(m)
 
    # Estaciones Transformadoras
    for nombre, lat, lon in ESTACIONES_TRANSFORMADORAS:
        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color="orange", icon="bolt", prefix="fa"),
            popup=f"Estación Transformadora: {nombre}",
        ).add_to(m)
 
    # Puertos
    for nombre, lat, lon in PUERTOS:
        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
            popup=f"Puerto: {nombre}",
        ).add_to(m)
 
    # Aeropuertos
    for nombre, lat, lon in AEROPUERTOS:
        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color="cadetblue", icon="plane", prefix="fa"),
            popup=f"Aeropuerto: {nombre}",
        ).add_to(m)
 
    # Últimas ubicaciones evaluadas por la población (para visualizar dispersión)
    for ind in poblacion_final:
        p = interpolate_point(ind["t"], BOUNDARY_POINTS)
        folium.CircleMarker(
            location=p, radius=2, color="gray", fill=True,
            fill_opacity=0.4,
        ).add_to(m)
 
    # Anillos de restricción poblacional alrededor del punto óptimo
    folium.Circle(
        location=punto_optimo, radius=RADIO_ZONA_BAJA_KM * 1000,
        color="#c62828", weight=2, fill=True, fill_opacity=0.05,
        tooltip="Zona 0-2 millas: densidad debe ser muy baja",
    ).add_to(m)
    folium.Circle(
        location=punto_optimo, radius=RADIO_ZONA_MEDIA_KM * 1000,
        color="#f9a825", weight=2, fill=True, fill_opacity=0.03,
        tooltip="Zona 2-10 millas: densidad media tolerada",
    ).add_to(m)
 
    # Punto óptimo
    detalle = mejor["detalle"]
    cumple = detalle.get("cumple_restriccion_dura", True)
    estado_restriccion = (
        "✅ CUMPLE restricción dura (0-2 mi)" if cumple
        else "⚠️ VIOLA restricción dura (0-2 mi)"
    )
    popup_html = (
        f"<b>Ubicación óptima SMR</b><br>"
        f"Lat: {punto_optimo[0]:.4f}, Lon: {punto_optimo[1]:.4f}<br>"
        f"<hr>"
        f"Fitness costo construcción: {detalle['f_costo']:.3f}<br>"
        f"&nbsp;&nbsp;Costo estimado: USD {detalle['costo_construccion_usd']:,.0f}<br>"
        f"Fitness poblacional: {detalle['f_poblacional']:.3f}<br>"
        f"&nbsp;&nbsp;Densidad máx. 0-2 mi: {detalle['dens_zona_baja']:.1f} hab/km²<br>"
        f"&nbsp;&nbsp;Densidad máx. 2-10 mi: {detalle['dens_zona_media']:.1f} hab/km²<br>"
        f"&nbsp;&nbsp;{estado_restriccion}<br>"
        f"<hr>"
        f"<b>Fitness total: {fitness_total:.3f}</b>"
    ).replace(",", ".")
    folium.Marker(
        location=punto_optimo,
        icon=folium.Icon(color="red" if cumple else "black", icon="star", prefix="fa"),
        popup=folium.Popup(popup_html, max_width=320),
        tooltip="Ubicación óptima",
    ).add_to(m)
 
    folium.LayerControl().add_to(m)
 
    m.save(path)
 
 
 
# MAIN
 
 
if __name__ == "__main__":
    # --- Sin seed fija: cada corrida explora la aleatoriedad de forma
    #     distinta. Si en algún momento necesitás reproducibilidad exacta
    #     de una corrida puntual, descomentá random.seed(N) / np.random.seed(N).
    random.seed()
    np.random.seed()
 
    mejor, historia_mejor, historia_promedio, historia_peor, historia_std, poblacion_final = correr_ga()
    mejor_fitness_total = historia_mejor[-1]
 
    print("=== RESULTADO ===")
    print(f"t óptimo (posición en la línea frontera): {mejor['t']:.4f}")
    print(f"Punto óptimo (lat, lon): {mejor['detalle']['punto']}")
    print(f"Fitness costo construcción: {mejor['detalle']['f_costo']:.3f}  "
          f"(USD {mejor['detalle']['costo_construccion_usd']:,.0f})")
    print(f"Fitness poblacional:        {mejor['detalle']['f_poblacional']:.3f}  "
          f"(dens. 0-2mi: {mejor['detalle']['dens_zona_baja']:.1f} hab/km², "
          f"dens. 2-10mi: {mejor['detalle']['dens_zona_media']:.1f} hab/km²)")
    print(f"Fitness total:              {mejor_fitness_total:.3f}")
 
    # --- Carpeta y nombres de archivo con timestamp para no pisar
    #     corridas anteriores y poder comparar resultados entre sí.
    OUTPUT_DIR = "resultados_ga"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    path_mapa = os.path.join(OUTPUT_DIR, f"mapa_optimo_{timestamp}.html")
    path_grafico = os.path.join(OUTPUT_DIR, f"convergencia_ga_{timestamp}.png")
    path_json = os.path.join(OUTPUT_DIR, f"resultado_optimo_{timestamp}.json")
 
    graficar_convergencia(historia_mejor, historia_promedio, historia_peor, historia_std, path=path_grafico)
    generar_mapa(mejor, poblacion_final, mejor_fitness_total, path=path_mapa)
 
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "t": mejor["t"],
            "punto": mejor["detalle"]["punto"],
            "fitness_costo": mejor["detalle"]["f_costo"],
            "costo_construccion_usd": mejor["detalle"]["costo_construccion_usd"],
            "fitness_poblacional": mejor["detalle"]["f_poblacional"],
            "densidad_zona_0_2mi": mejor["detalle"]["dens_zona_baja"],
            "densidad_zona_2_10mi": mejor["detalle"]["dens_zona_media"],
            "fitness_total": mejor_fitness_total,
        }, f, ensure_ascii=False, indent=2)
 
    # --- Log acumulado de todas las corridas, para comparar resultados
    #     entre sí sin tener que abrir cada JSON individual.
    path_log = os.path.join(OUTPUT_DIR, "log_corridas.csv")
    existe_log = os.path.exists(path_log)
    with open(path_log, "a", encoding="utf-8") as f:
        if not existe_log:
            f.write("timestamp,t,lat,lon,fitness_costo,costo_construccion_usd,"
                    "fitness_poblacional,densidad_0_2mi,densidad_2_10mi,"
                    "fitness_mejor,fitness_promedio,fitness_peor,fitness_std\n")
        lat, lon = mejor["detalle"]["punto"]
        f.write(f"{timestamp},{mejor['t']:.6f},{lat:.6f},{lon:.6f},"
                f"{mejor['detalle']['f_costo']:.4f},{mejor['detalle']['costo_construccion_usd']:.2f},"
                f"{mejor['detalle']['f_poblacional']:.4f},"
                f"{mejor['detalle']['dens_zona_baja']:.2f},{mejor['detalle']['dens_zona_media']:.2f},"
                f"{mejor_fitness_total:.4f},"
                f"{historia_promedio[-1]:.4f},{historia_peor[-1]:.4f},{historia_std[-1]:.4f}\n")
 
    print(f"\nArchivos generados en '{OUTPUT_DIR}/':")
    print(f"  - {path_mapa}")
    print(f"  - {path_grafico}")
    print(f"  - {path_json}")
    print(f"  - {path_log} (acumula todas las corridas)")