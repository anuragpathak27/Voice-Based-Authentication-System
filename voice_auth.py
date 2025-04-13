# IMPORT SYSTEM FILES
import argparse
import os
import numpy as np
import warnings
import tensorflow as tf
from scipy.spatial.distance import euclidean
import logging

# Suppress warnings and logging
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# IMPORT USER-DEFINED FUNCTIONS
from feature_extraction import get_embedding, get_embeddings_from_list_file
import parameters as p

# Set the model directory path
p.MODEL_FILE = 'voice_auth_model_cnn'  # Point to the directory, not the file
print("Model directory path:", p.MODEL_FILE)

# Check if the model directory exists
if not os.path.exists(p.MODEL_FILE):
    print(f"Error: The directory {p.MODEL_FILE} does not exist.")
    exit()

# Check if the saved_model.pb file exists
if not os.path.exists(os.path.join(p.MODEL_FILE, 'saved_model.pb')):
    print(f"Error: The file saved_model.pb does not exist in {p.MODEL_FILE}.")
    exit()

# args() returns the args passed to the script
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', help='Task to do. Either "enroll" or "recognize"', required=True)
    parser.add_argument('-n', '--name', help='Specify the name of the person you want to enroll', required=False)
    parser.add_argument('-f', '--file', help='Specify the audio file you want to enroll', required=True)
    return parser.parse_args()

def enroll(name, file):
    """Enroll a user with an audio file"""
    print("Loading model weights from [{}]....".format(p.MODEL_FILE))
    try:
        model = tf.saved_model.load(p.MODEL_FILE)
        predict_fn = model.signatures['serving_default']
    except Exception as e:
        print(f"Failed to load weights from the weights file: {e}")
        exit()
    
    try:
        print("Processing enroll sample....")
        enroll_result = get_embedding(model, file, p.MAX_SEC)
        enroll_embs = np.array(enroll_result.tolist())
        speaker = name
    except Exception as e:
        print(f"Error processing the input audio file: {e}")
        return
    
    try:
        np.save(os.path.join(p.EMBED_LIST_FILE, f"{speaker}.npy"), enroll_embs)
        print("Successfully enrolled the user")
    except Exception as e:
        print(f"Unable to save the user into the database: {e}")

def enroll_csv(csv_file):
    """Enroll a list of users using a CSV file"""
    print("Getting the model weights from [{}]".format(p.MODEL_FILE))
    try:
        model = tf.saved_model.load(p.MODEL_FILE)
    except Exception as e:
        print(f"Failed to load weights from the weights file: {e}")
        exit()
    
    print("Processing enroll samples....")
    try:
        enroll_results = get_embeddings_from_list_file(model, csv_file, p.MAX_SEC)
        enroll_embs = np.array([emb.tolist() for emb in enroll_results['embedding']])
        speakers = enroll_results['speaker']
    except Exception as e:
        print(f"Error processing the input audio files: {e}")
        return
    
    try:
        for i, speaker in enumerate(speakers):
            np.save(os.path.join(p.EMBED_LIST_FILE, f"{speaker}.npy"), enroll_embs[i])
            print(f"Successfully enrolled the user: {speaker}")
    except Exception as e:
        print(f"Unable to save the user into the database: {e}")

def recognize(file):
    """Recognize the input audio file by comparing to saved users' voice prints"""
    if os.path.exists(p.EMBED_LIST_FILE):
        embeds = os.listdir(p.EMBED_LIST_FILE)
    if len(embeds) == 0:
        print("No enrolled users found")
        exit()
    
    print("Loading model weights from [{}]....".format(p.MODEL_FILE))
    try:
        model = tf.saved_model.load(p.MODEL_FILE)
    except Exception as e:
        print(f"Failed to load weights from the weights file: {e}")
        exit()
    
    distances = {}
    print("Processing test sample....")
    print("Comparing test sample against enroll samples....")
    try:
        test_result = get_embedding(model, file, p.MAX_SEC)
        test_embs = np.array(test_result.tolist())
    except Exception as e:
        print(f"Error processing the test audio file: {e}")
        return
    
    for emb in embeds:
        try:
            enroll_embs = np.load(os.path.join(p.EMBED_LIST_FILE, emb))
            speaker = emb.replace(".npy", "")
            distance = euclidean(test_embs, enroll_embs)
            distances.update({speaker: distance})
        except Exception as e:
            print(f"Error loading or comparing embeddings for {emb}: {e}")
    
    if distances and min(distances.values()) < p.THRESHOLD:
        print("Recognized:", min(distances, key=distances.get))
    else:
        print("Could not identify the user, try enrolling again with a clear voice sample")
        print("Score:", min(distances.values()) if distances else "N/A")

# Helper function to get file extension
def get_extension(filename):
    """Extract the file extension from a filename."""
    return os.path.splitext(filename)[1][1:].lower()

if __name__ == '__main__':
    try:
        args = args()
    except Exception as e:
        print(f"An exception occurred: {e}")
        exit()
    
    task = args.task
    file = args.file
    name = args.name if args.name else None

    if get_extension(file) == 'csv':
        if task == 'enroll':
            enroll_csv(file)
        elif task == 'recognize':
            print("Recognize argument cannot process a comma-separated file. Please specify an audio file.")
    else:
        if task == 'enroll':
            if not name:
                print("Missing argument: -n name is required for the user name")
                exit()
            enroll(name, file)
        elif task == 'recognize':
            recognize(file)