import os
import config
import random
import shutil

examples = os.listdir(os.path.join(config.DATA_DIR, "train"))
val_examples = random.sample(examples, 48)

def mover_archivos(origen, destino):
    # Verificar si la carpeta de origen existe
    if not os.path.exists(origen):
        print(f"La carpeta de origen '{origen}' no existe.")
        return

    # Verificar si la carpeta de destino existe, si no, crearla
    if not os.path.exists(destino):
        os.makedirs(destino)
        print(f"La carpeta de destino '{destino}' no existía y fue creada.")

    # Obtener la lista de archivos en la carpeta de origen
    archivos = val_examples

    # Mover cada archivo a la carpeta de destino
    for archivo in archivos:
        # Mover la imagen
        ruta_origen = os.path.join(origen, archivo)
        ruta_destino = os.path.join(destino, archivo)
        shutil.move(ruta_origen, ruta_destino)
        # Mover la máscara
        ruta_origen = os.path.join("data/train_masks", archivo.replace(".jpg", "_mask.gif"))
        ruta_destino = os.path.join("data/validation_masks", archivo.replace(".jpg", "_masks.gif"))
        shutil.move(ruta_origen, ruta_destino)


if __name__=="__main__":
    # Definir las rutas de origen y destino
    carpeta_origen = "data/train"
    carpeta_destino = "data/validation"

    # Llamar a la función para mover archivos
    mover_archivos(carpeta_origen, carpeta_destino)
