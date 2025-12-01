import socket
import json
import subprocess
import getpass 
import time
import threading

# Constantes définies
PUSCH_SEUIL = -26
PCI_VALEUR1 = 7
CELL_VALEUR1= 0x66C001
PCI_VALEUR2 = 8
CELL_VALEUR2= 0x66C002

UDP_IP = "127.0.0.1"
UDP_PORT = 55555

# Gestion du délai entre deux handovers
LAST_HO_TIME = 0
HO_DELAY_SECONDS = 3  # ⏱️ délai minimum entre deux handovers

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Bind the socket to the IP address and port
sock.bind((UDP_IP, UDP_PORT))

print("UDP Receiver started...")

received_data = []
extracted_metrics = []

def check_na_window():
    """
    Vérifie si on a plus de 2 occurrences N/A dans les 3 dernières secondes
    """
    pass

def handover_decision(pci):
    """
    Prend la décision de handover basée sur le PCI
    """
    if pci == PCI_VALEUR1:
        print("Décision: Handover de cell 7 vers 8")
        execute_handover_command(CELL_VALEUR2)
    elif pci == PCI_VALEUR2:
        print("Décision: Handover de cell 8 vers 7")
        execute_handover_command(CELL_VALEUR1)
    else:
        print(f"PCI {pci} non reconnu pour le handover")

def execute_handover_command(target_cell):
    """
    Lance la commande de handover dans un nouveau terminal avec une temporisation minimale
    """
    global LAST_HO_TIME

    current_time = time.time()
    time_since_last_ho = current_time - LAST_HO_TIME

    # Vérifier si la temporisation est respectée
    if time_since_last_ho < HO_DELAY_SECONDS:
        print(f"⚠️ Handover ignoré : dernier HO exécuté il y a {time_since_last_ho:.2f}s (< {HO_DELAY_SECONDS}s).")
        return

    LAST_HO_TIME = current_time

    print(f"Tentative d'exécution de HO vers {target_cell}...")

    docker_cmd = (
        f"docker compose exec python_xapp_runner "
        f"./simple_rc_ho_xapp.py "
        f"--e2_node_id gnb_001_001_00019b "
        f"--plmn 00101 "
        f"--amf_ue_ngap_id 1 "
        f"--target_nr_cell_id {target_cell}"
    )

    try:
        subprocess.Popen([
            "gnome-terminal",
            "--", "bash", "-c",
            f"{docker_cmd}; exec bash"
        ])
        print(f"✅ Nouveau terminal lancé avec la commande pour la cellule {target_cell}.")
    except Exception as e:
        print(f"❌ Erreur lors du lancement du handover : {e}")

def process_metrics(pci, pusch_rsrp_db):
    """
    Vérifie la valeur du RSRP et déclenche le handover si nécessaire.
    """
    if pusch_rsrp_db != "n/a":
        try:
            rsrp_val = float(pusch_rsrp_db)
            if rsrp_val <= PUSCH_SEUIL:
                handover_decision(pci)
        except ValueError:
            print(f"⚠️ Valeur RSRP non numérique reçue : {pusch_rsrp_db}")

try:
    while True:
        # 1. Données UDP
        data, addr = sock.recvfrom(3000)
        
        try:
            # 2. Décodage JSON
            json_data = json.loads(data.decode('utf-8'))
            print("📨 Données JSON reçues:", json_data)
            received_data.append(json_data)
            
            if 'ue_list' in json_data and json_data['ue_list']:
                for ue in json_data['ue_list']:
                    if 'ue_container' in ue:
                        ue_container = ue['ue_container']
                        pci = ue_container.get('pci')
                        pusch_rsrp_db = ue_container.get('pusch_rsrp_db')
                        timestamp = json_data.get('timestamp')
                        
                        # 3. Extraction métriques
                        if pci is not None and pusch_rsrp_db is not None:
                            extracted_metric = {
                                'timestamp': timestamp,
                                'pci': pci,
                                'pusch_rsrp_db': pusch_rsrp_db
                            }
                            extracted_metrics.append(extracted_metric)
                            print(f"Métriques extraites - PCI: {pci}, PUSCH RSRP: {pusch_rsrp_db} dB")
                            
                            # 4. Vérification seuil avec nouvelles conditions
                            process_metrics(pci, pusch_rsrp_db)
            
        except json.JSONDecodeError:
            print("Données reçues non au format JSON:", data.decode('utf-8'))

except KeyboardInterrupt:
    # 7. Sauvegarde données
    filename_raw = "gnb_metrics.json"
    with open(filename_raw, "w") as file:
        for entry in received_data:
            json.dump(entry, file)
            file.write("\n")
    print(f"Données brutes sauvegardées dans {filename_raw}")
    
    filename_extracted = "extracted_pci_rsrp_metrics.json"
    with open(filename_extracted, "w") as file:
        for entry in extracted_metrics:
            json.dump(entry, file)
            file.write("\n")
    print(f"Métriques extraites sauvegardées dans {filename_extracted}")
