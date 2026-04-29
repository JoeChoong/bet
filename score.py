
import os
import re
import pandas as pd
import joblib
import numpy as np
import shap
#from sklearn.inspection import partial_dependence
import json
import pickle
# connection library
from io import StringIO
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

model = None

# AML model init
# def init():
#     global model
#     base_dir = Path(__file__).resolve().parent
#     candidate_paths = [
#         base_dir / "lgb_cat_model_v8.pkl",
#         base_dir / "src" / "lgb_cat_model_v8.pkl",
#     ]

#     for model_path in candidate_paths:
#         if model_path.exists():
#             model = joblib.load(model_path)
#             return

#     raise FileNotFoundError("Could not find lgb_cat_model_v8.pkl in project root or src/")

# def __init__(self):
#     global model
    
#     model = joblib.load("C:/Users/I008328/lgb_cat_model_v8.pkl")

with open("lgb_cat_model_v8.pkl","rb") as f:
    model = pickle.load(f)

# Return model predict result
def score(input_json):

    # Mapping between questions codes
    full = {'Question_code': {0: 'Q055_A',
        1: 'Q055_B',
        2: 'Q055_C',
        3: 'Q055_D',
        4: 'Q055_E',
        5: 'Q055_F',
        6: 'Q056_A',
        7: 'Q056_B',
        8: 'Q056_C',
        9: 'Q056_D',
        10: 'Q056_E',
        11: 'Q056_F',
        12: 'Q056_G',
        13: 'Q056_H',
        14: 'Q056_I',
        15: 'Q056_J',
        16: 'Q056_K',
        17: 'Q056_L',
        18: 'Q060_A',
        19: 'Q060_B',
        20: 'Q060_C',
        21: 'Q060_D',
        22: 'Q060_E',
        23: 'Q060_F',
        24: 'Q060_G',
        25: 'Q060_H',
        26: 'Q060_I',
        27: 'Q057_A',
        28: 'Q057_B',
        29: 'Q057_C',
        30: 'Q057_D',
        31: 'Q057_E',
        32: 'Q057_F',
        33: 'Q058_A',
        34: 'Q058_B',
        35: 'Q058_C',
        36: 'Q058_D',
        37: 'Q058_E',
        38: 'Q058_F',
        39: 'Q050_A',
        40: 'Q050_B',
        41: 'Q050_C',
        42: 'Q050_D',
        43: 'Q050_E',
        44: 'Q050_F',
        45: 'Q051_A',
        46: 'Q051_B',
        47: 'Q051_C',
        48: 'Q051_D',
        49: 'Q051_E',
        50: 'Q051_F',
        51: 'Q052_I',
        52: 'Q052_A',
        53: 'Q052_B',
        54: 'Q052_C',
        55: 'Q052_D',
        56: 'Q052_E',
        57: 'Q052_F',
        58: 'Q052_G',
        59: 'Q052_H',
        60: 'Q053_A',
        61: 'Q053_B',
        62: 'Q053_C',
        63: 'Q053_D',
        64: 'Q053_E',
        65: 'Q053_F',
        66: 'Q054_A',
        67: 'Q054_B',
        68: 'Q054_C',
        69: 'Q054_D',
        70: 'Q054_E',
        71: 'Q066_A',
        72: 'Q066_B',
        73: 'Q066_C',
        74: 'Q067_A',
        75: 'Q067_B',
        76: 'Q067_C',
        88: 'Q061_A',
        89: 'Q061_B',
        90: 'Q061_C',
        91: 'Q061_D',
        92: 'Q061_E',
        93: 'Q061_F',
        94: 'Q062_A',
        95: 'Q062_B',
        96: 'Q062_C',
        97: 'Q062_D',
        98: 'Q062_E',
        99: 'Q064_A',
        100: 'Q064_B',
        101: 'Q064_C',
        102: 'Q064_D',
        103: 'Q065_A',
        104: 'Q065_B',
        105: 'Q065_C',
        106: 'Q065_D',
        107: 'Q065_E',
        108: 'Q065_F',
        109: 'Q070_A',
        110: 'Q070_B',
        111: 'Q070_C',
        112: 'Q070_D',
        113: 'Q070_E',
        114: 'Q070_F',
        115: 'Q070_G',
        116: 'Q070_H',
        117: 'Q070_I',
        118: 'Q070_J',
        119: 'Q070_K',
        120: 'Q070_L',
        121: 'Q071_A',
        122: 'Q071_B',
        123: 'Q071_C',
        124: 'Q071_D',
        125: 'Q071_E',
        126: 'Q071_F',
        127: 'Q071_G',
        128: 'Q071_H',
        129: 'Q076_A',
        130: 'Q076_B',
        131: 'Q076_C',
        132: 'Q076_D',
        133: 'Q076_E',
        134: 'Q076_F',
        135: 'Q076_G',
        136: 'Q076_H',
        137: 'Q076_I',
        138: 'Q076_J',
        139: 'Q076_K',
        140: 'Q076_L',
        141: 'Q001_A',
        142: 'Q001_B',
        143: 'Q001_C',
        144: 'Q002_A',
        145: 'Q002_B',
        146: 'Q002_C',
        147: 'Q003_A',
        148: 'Q003_B',
        149: 'Q003_C',
        150: 'Q072_A',
        151: 'Q072_B',
        152: 'Q072_C',
        153: 'Q005_A',
        154: 'Q005_B',
        155: 'Q005_C',
        156: 'Q006_A',
        157: 'Q006_B',
        158: 'Q006_C',
        159: 'Q007_A',
        160: 'Q007_B',
        161: 'Q007_C',
        162: 'Q010_A',
        163: 'Q010_B',
        164: 'Q010_C',
        165: 'Q011_A',
        166: 'Q011_B',
        167: 'Q011_C',
        168: 'Q012_A',
        169: 'Q012_B',
        170: 'Q012_C',
        171: 'Q013_A',
        172: 'Q013_B',
        173: 'Q013_C',
        174: 'Q014_A',
        175: 'Q014_B',
        176: 'Q014_C',
        177: 'Q015_A',
        178: 'Q015_B',
        179: 'Q015_C',
        180: 'Q018_A',
        181: 'Q018_B',
        182: 'Q018_C',
        183: 'Q019_A',
        184: 'Q019_B',
        185: 'Q019_C',
        186: 'Q020_A',
        187: 'Q020_B',
        188: 'Q020_C',
        189: 'Q021_A',
        190: 'Q021_B',
        191: 'Q021_C',
        192: 'Q022_A',
        193: 'Q022_B',
        194: 'Q022_C',
        195: 'Q025_A',
        196: 'Q025_B',
        197: 'Q025_C',
        198: 'Q026_A',
        199: 'Q026_B',
        200: 'Q026_C',
        201: 'Q028_A',
        202: 'Q028_B',
        203: 'Q028_C',
        204: 'Q029_A',
        205: 'Q029_B',
        206: 'Q029_C',
        207: 'Q031_A',
        208: 'Q031_B',
        209: 'Q031_C',
        210: 'Q032_A',
        211: 'Q032_B',
        212: 'Q032_C',
        213: 'Q034_A',
        214: 'Q034_B',
        215: 'Q034_C',
        216: 'Q035_A',
        217: 'Q035_B',
        218: 'Q035_C',
        219: 'Q036_A',
        220: 'Q036_B',
        221: 'Q036_C',
        222: 'Q038_A',
        223: 'Q038_B',
        224: 'Q038_C',
        225: 'Q039_A',
        226: 'Q039_B',
        227: 'Q039_C',
        228: 'Q040_A',
        229: 'Q040_B',
        230: 'Q040_C',
        231: 'Q041_A',
        232: 'Q041_B',
        233: 'Q041_C',
        234: 'Q042_A',
        235: 'Q042_B',
        236: 'Q042_C',
        237: 'Q043_A',
        238: 'Q043_B',
        239: 'Q043_C',
        240: 'Q073_A',
        241: 'Q073_B',
        242: 'Q073_C',
        243: 'Q074_A',
        244: 'Q074_B',
        245: 'Q074_C',
        246: 'Q075_A',
        247: 'Q075_B',
        248: 'Q075_C'},
        'Question_AnswerCode': {0: 'BQUE0001_BANS0001',
        1: 'BQUE0001_BANS0002',
        2: 'BQUE0001_BANS0003',
        3: 'BQUE0001_BANS0004',
        4: 'BQUE0001_BANS0005',
        5: 'BQUE0001_BANS0006',
        6: 'BQUE0002_BANS0001',
        7: 'BQUE0002_BANS0002',
        8: 'BQUE0002_BANS0003',
        9: 'BQUE0002_BANS0004',
        10: 'BQUE0002_BANS0005',
        11: 'BQUE0002_BANS0006',
        12: 'BQUE0002_BANS0007',
        13: 'BQUE0002_BANS0008',
        14: 'BQUE0002_BANS0009',
        15: 'BQUE0002_BANS0010',
        16: 'BQUE0002_BANS0011',
        17: 'BQUE0002_BANS0012',
        18: 'BQUE0003_BANS0001',
        19: 'BQUE0003_BANS0002',
        20: 'BQUE0003_BANS0003',
        21: 'BQUE0003_BANS0004',
        22: 'BQUE0003_BANS0005',
        23: 'BQUE0003_BANS0006',
        24: 'BQUE0003_BANS0007',
        25: 'BQUE0003_BANS0008',
        26: 'BQUE0003_BANS0009',
        27: 'BQUE0004_BANS0001',
        28: 'BQUE0004_BANS0002',
        29: 'BQUE0004_BANS0003',
        30: 'BQUE0004_BANS0004',
        31: 'BQUE0004_BANS0005',
        32: 'BQUE0004_BANS0006',
        33: 'BQUE0005_BANS0001',
        34: 'BQUE0005_BANS0002',
        35: 'BQUE0005_BANS0003',
        36: 'BQUE0005_BANS0004',
        37: 'BQUE0005_BANS0005',
        38: 'BQUE0005_BANS0006',
        39: 'BQUE0006_BANS0001',
        40: 'BQUE0006_BANS0002',
        41: 'BQUE0006_BANS0003',
        42: 'BQUE0006_BANS0004',
        43: 'BQUE0006_BANS0005',
        44: 'BQUE0006_BANS0006',
        45: 'BQUE0007_BANS0001',
        46: 'BQUE0007_BANS0002',
        47: 'BQUE0007_BANS0003',
        48: 'BQUE0007_BANS0004',
        49: 'BQUE0007_BANS0005',
        50: 'BQUE0007_BANS0006',
        51: 'BQUE0008_BANS0001',
        52: 'BQUE0008_BANS0002',
        53: 'BQUE0008_BANS0003',
        54: 'BQUE0008_BANS0004',
        55: 'BQUE0008_BANS0005',
        56: 'BQUE0008_BANS0006',
        57: 'BQUE0008_BANS0007',
        58: 'BQUE0008_BANS0008',
        59: 'BQUE0008_BANS0009',
        60: 'BQUE0009_BANS0001',
        61: 'BQUE0009_BANS0002',
        62: 'BQUE0009_BANS0003',
        63: 'BQUE0009_BANS0004',
        64: 'BQUE0009_BANS0005',
        65: 'BQUE0009_BANS0006',
        66: 'BQUE0010_BANS0001',
        67: 'BQUE0010_BANS0002',
        68: 'BQUE0010_BANS0003',
        69: 'BQUE0010_BANS0004',
        70: 'BQUE0010_BANS0005',
        71: 'BQUE0011_BANS0001',
        72: 'BQUE0011_BANS0002',
        73: 'BQUE0011_BANS0003',
        74: 'BQUE0012_BANS0001',
        75: 'BQUE0012_BANS0002',
        76: 'BQUE0012_BANS0003',
        88: 'BQUE0014_BANS0001',
        89: 'BQUE0014_BANS0002',
        90: 'BQUE0014_BANS0003',
        91: 'BQUE0014_BANS0004',
        92: 'BQUE0014_BANS0005',
        93: 'BQUE0014_BANS0006',
        94: 'BQUE0015_BANS0001',
        95: 'BQUE0015_BANS0002',
        96: 'BQUE0015_BANS0003',
        97: 'BQUE0015_BANS0004',
        98: 'BQUE0015_BANS0005',
        99: 'BQUE0016_BANS0001',
        100: 'BQUE0016_BANS0002',
        101: 'BQUE0016_BANS0003',
        102: 'BQUE0016_BANS0004',
        103: 'BQUE0017_BANS0001',
        104: 'BQUE0017_BANS0002',
        105: 'BQUE0017_BANS0003',
        106: 'BQUE0017_BANS0004',
        107: 'BQUE0017_BANS0005',
        108: 'BQUE0017_BANS0006',
        109: 'BQUE0018_BANS0001',
        110: 'BQUE0018_BANS0002',
        111: 'BQUE0018_BANS0003',
        112: 'BQUE0018_BANS0004',
        113: 'BQUE0018_BANS0005',
        114: 'BQUE0018_BANS0006',
        115: 'BQUE0018_BANS0007',
        116: 'BQUE0018_BANS0008',
        117: 'BQUE0018_BANS0009',
        118: 'BQUE0018_BANS0010',
        119: 'BQUE0018_BANS0011',
        120: 'BQUE0018_BANS0012',
        121: 'BQUE0019_BANS0001',
        122: 'BQUE0019_BANS0002',
        123: 'BQUE0019_BANS0003',
        124: 'BQUE0019_BANS0004',
        125: 'BQUE0019_BANS0005',
        126: 'BQUE0019_BANS0006',
        127: 'BQUE0019_BANS0007',
        128: 'BQUE0019_BANS0008',
        129: 'BQUE0020_BANS0001',
        130: 'BQUE0020_BANS0002',
        131: 'BQUE0020_BANS0003',
        132: 'BQUE0020_BANS0004',
        133: 'BQUE0020_BANS0005',
        134: 'BQUE0020_BANS0006',
        135: 'BQUE0020_BANS0007',
        136: 'BQUE0020_BANS0008',
        137: 'BQUE0020_BANS0009',
        138: 'BQUE0020_BANS0010',
        139: 'BQUE0020_BANS0011',
        140: 'BQUE0020_BANS0012',
        141: 'QUE0001_ANS0001',
        142: 'QUE0001_ANS0002',
        143: 'QUE0001_ANS0003',
        144: 'QUE0002_ANS0001',
        145: 'QUE0002_ANS0002',
        146: 'QUE0002_ANS0003',
        147: 'QUE0003_ANS0001',
        148: 'QUE0003_ANS0002',
        149: 'QUE0003_ANS0003',
        150: 'QUE0004_ANS0001',
        151: 'QUE0004_ANS0002',
        152: 'QUE0004_ANS0003',
        153: 'QUE0005_ANS0001',
        154: 'QUE0005_ANS0002',
        155: 'QUE0005_ANS0003',
        156: 'QUE0006_ANS0001',
        157: 'QUE0006_ANS0002',
        158: 'QUE0006_ANS0003',
        159: 'QUE0007_ANS0001',
        160: 'QUE0007_ANS0002',
        161: 'QUE0007_ANS0003',
        162: 'QUE0008_ANS0001',
        163: 'QUE0008_ANS0002',
        164: 'QUE0008_ANS0003',
        165: 'QUE0009_ANS0001',
        166: 'QUE0009_ANS0002',
        167: 'QUE0009_ANS0003',
        168: 'QUE0010_ANS0001',
        169: 'QUE0010_ANS0002',
        170: 'QUE0010_ANS0003',
        171: 'QUE0011_ANS0001',
        172: 'QUE0011_ANS0002',
        173: 'QUE0011_ANS0003',
        174: 'QUE0012_ANS0001',
        175: 'QUE0012_ANS0002',
        176: 'QUE0012_ANS0003',
        177: 'QUE0013_ANS0001',
        178: 'QUE0013_ANS0002',
        179: 'QUE0013_ANS0003',
        180: 'QUE0014_ANS0001',
        181: 'QUE0014_ANS0002',
        182: 'QUE0014_ANS0003',
        183: 'QUE0015_ANS0001',
        184: 'QUE0015_ANS0002',
        185: 'QUE0015_ANS0003',
        186: 'QUE0016_ANS0001',
        187: 'QUE0016_ANS0002',
        188: 'QUE0016_ANS0003',
        189: 'QUE0017_ANS0001',
        190: 'QUE0017_ANS0002',
        191: 'QUE0017_ANS0003',
        192: 'QUE0018_ANS0001',
        193: 'QUE0018_ANS0002',
        194: 'QUE0018_ANS0003',
        195: 'QUE0019_ANS0001',
        196: 'QUE0019_ANS0002',
        197: 'QUE0019_ANS0003',
        198: 'QUE0020_ANS0001',
        199: 'QUE0020_ANS0002',
        200: 'QUE0020_ANS0003',
        201: 'QUE0021_ANS0001',
        202: 'QUE0021_ANS0002',
        203: 'QUE0021_ANS0003',
        204: 'QUE0022_ANS0001',
        205: 'QUE0022_ANS0002',
        206: 'QUE0022_ANS0003',
        207: 'QUE0023_ANS0001',
        208: 'QUE0023_ANS0002',
        209: 'QUE0023_ANS0003',
        210: 'QUE0024_ANS0001',
        211: 'QUE0024_ANS0002',
        212: 'QUE0024_ANS0003',
        213: 'QUE0025_ANS0001',
        214: 'QUE0025_ANS0002',
        215: 'QUE0025_ANS0003',
        216: 'QUE0026_ANS0001',
        217: 'QUE0026_ANS0002',
        218: 'QUE0026_ANS0003',
        219: 'QUE0027_ANS0001',
        220: 'QUE0027_ANS0002',
        221: 'QUE0027_ANS0003',
        222: 'QUE0028_ANS0001',
        223: 'QUE0028_ANS0002',
        224: 'QUE0028_ANS0003',
        225: 'QUE0029_ANS0001',
        226: 'QUE0029_ANS0002',
        227: 'QUE0029_ANS0003',
        228: 'QUE0030_ANS0001',
        229: 'QUE0030_ANS0002',
        230: 'QUE0030_ANS0003',
        231: 'QUE0031_ANS0001',
        232: 'QUE0031_ANS0002',
        233: 'QUE0031_ANS0003',
        234: 'QUE0032_ANS0001',
        235: 'QUE0032_ANS0002',
        236: 'QUE0032_ANS0003',
        237: 'QUE0033_ANS0001',
        238: 'QUE0033_ANS0002',
        239: 'QUE0033_ANS0003',
        240: 'QUE0034_ANS0001',
        241: 'QUE0034_ANS0002',
        242: 'QUE0034_ANS0003',
        243: 'QUE0035_ANS0001',
        244: 'QUE0035_ANS0002',
        245: 'QUE0035_ANS0003',
        246: 'QUE0036_ANS0001',
        247: 'QUE0036_ANS0002',
        248: 'QUE0036_ANS0003'},
        'Category': {0: 'Others',
        1: 'Others',
        2: 'Others',
        3: 'Others',
        4: 'Others',
        5: 'Others',
        6: 'Others',
        7: 'Others',
        8: 'Others',
        9: 'Others',
        10: 'Others',
        11: 'Others',
        12: 'Others',
        13: 'Others',
        14: 'Others',
        15: 'Others',
        16: 'Others',
        17: 'Others',
        18: 'Others',
        19: 'Others',
        20: 'Others',
        21: 'Others',
        22: 'Others',
        23: 'Others',
        24: 'Others',
        25: 'Others',
        26: 'Others',
        27: 'Others',
        28: 'Others',
        29: 'Others',
        30: 'Others',
        31: 'Others',
        32: 'Others',
        33: 'Others',
        34: 'Others',
        35: 'Others',
        36: 'Others',
        37: 'Others',
        38: 'Others',
        39: 'Others',
        40: 'Others',
        41: 'Others',
        42: 'Others',
        43: 'Others',
        44: 'Others',
        45: 'Others',
        46: 'Others',
        47: 'Others',
        48: 'Others',
        49: 'Others',
        50: 'Others',
        51: 'Others',
        52: 'Others',
        53: 'Others',
        54: 'Others',
        55: 'Others',
        56: 'Others',
        57: 'Others',
        58: 'Others',
        59: 'Others',
        60: 'Others',
        61: 'Others',
        62: 'Others',
        63: 'Others',
        64: 'Others',
        65: 'Others',
        66: 'Others',
        67: 'Others',
        68: 'Others',
        69: 'Others',
        70: 'Others',
        71: 'Others',
        72: 'Others',
        73: 'Others',
        74: 'Others',
        75: 'Others',
        76: 'Others',
        88: 'Network Resources',
        89: 'Network Resources',
        90: 'Network Resources',
        91: 'Network Resources',
        92: 'Network Resources',
        93: 'Network Resources',
        94: 'Network Resources',
        95: 'Network Resources',
        96: 'Network Resources',
        97: 'Network Resources',
        98: 'Network Resources',
        99: 'Network Resources',
        100: 'Network Resources',
        101: 'Network Resources',
        102: 'Network Resources',
        103: 'Network Resources',
        104: 'Network Resources',
        105: 'Network Resources',
        106: 'Network Resources',
        107: 'Network Resources',
        108: 'Network Resources',
        109: 'Others',
        110: 'Others',
        111: 'Others',
        112: 'Others',
        113: 'Others',
        114: 'Others',
        115: 'Others',
        116: 'Others',
        117: 'Others',
        118: 'Others',
        119: 'Others',
        120: 'Others',
        121: 'Others',
        122: 'Others',
        123: 'Others',
        124: 'Others',
        125: 'Others',
        126: 'Others',
        127: 'Others',
        128: 'Others',
        129: 'Others',
        130: 'Others',
        131: 'Others',
        132: 'Others',
        133: 'Others',
        134: 'Others',
        135: 'Others',
        136: 'Others',
        137: 'Others',
        138: 'Others',
        139: 'Others',
        140: 'Others',
        141: 'Self-conscientiousness',
        142: 'Self-conscientiousness',
        143: 'Self-conscientiousness',
        144: 'Self-conscientiousness',
        145: 'Self-conscientiousness',
        146: 'Self-conscientiousness',
        147: 'Self-conscientiousness',
        148: 'Self-conscientiousness',
        149: 'Self-conscientiousness',
        150: 'Self-conscientiousness',
        151: 'Self-conscientiousness',
        152: 'Self-conscientiousness',
        153: 'Self-conscientiousness',
        154: 'Self-conscientiousness',
        155: 'Self-conscientiousness',
        156: 'Self-conscientiousness',
        157: 'Self-conscientiousness',
        158: 'Self-conscientiousness',
        159: 'Curiosity',
        160: 'Curiosity',
        161: 'Curiosity',
        162: 'Curiosity',
        163: 'Curiosity',
        164: 'Curiosity',
        165: 'Curiosity',
        166: 'Curiosity',
        167: 'Curiosity',
        168: 'Curiosity',
        169: 'Curiosity',
        170: 'Curiosity',
        171: 'Curiosity',
        172: 'Curiosity',
        173: 'Curiosity',
        174: 'Ambition',
        175: 'Ambition',
        176: 'Ambition',
        177: 'Ambition',
        178: 'Ambition',
        179: 'Ambition',
        180: 'Ambition',
        181: 'Ambition',
        182: 'Ambition',
        183: 'Ambition',
        184: 'Ambition',
        185: 'Ambition',
        186: 'Ambition',
        187: 'Ambition',
        188: 'Ambition',
        189: 'Sociability',
        190: 'Sociability',
        191: 'Sociability',
        192: 'Sociability',
        193: 'Sociability',
        194: 'Sociability',
        195: 'Adaptability',
        196: 'Adaptability',
        197: 'Adaptability',
        198: 'Adaptability',
        199: 'Adaptability',
        200: 'Adaptability',
        201: 'Adaptability',
        202: 'Adaptability',
        203: 'Adaptability',
        204: 'Adaptability',
        205: 'Adaptability',
        206: 'Adaptability',
        207: 'Sociability',
        208: 'Sociability',
        209: 'Sociability',
        210: 'Sociability',
        211: 'Sociability',
        212: 'Sociability',
        213: 'Leadership',
        214: 'Leadership',
        215: 'Leadership',
        216: 'Leadership',
        217: 'Leadership',
        218: 'Leadership',
        219: 'Leadership',
        220: 'Leadership',
        221: 'Leadership',
        222: 'Leadership',
        223: 'Leadership',
        224: 'Leadership',
        225: 'Resilience',
        226: 'Resilience',
        227: 'Resilience',
        228: 'Resilience',
        229: 'Resilience',
        230: 'Resilience',
        231: 'Resilience',
        232: 'Resilience',
        233: 'Resilience',
        234: 'Resilience',
        235: 'Resilience',
        236: 'Resilience',
        237: 'Resilience',
        238: 'Resilience',
        239: 'Resilience',
        240: 'Compliance Awareness',
        241: 'Compliance Awareness',
        242: 'Compliance Awareness',
        243: 'Compliance Awareness',
        244: 'Compliance Awareness',
        245: 'Compliance Awareness',
        246: 'Compliance Awareness',
        247: 'Compliance Awareness',
        248: 'Compliance Awareness'},
        'Question_Answer_test': 
        {0: 'You highest academic qualification is:-Masters Degree, PhD or above',
        1: 'You highest academic qualification is:-Bachelors Degree (including incoming graduates)',
        2: 'You highest academic qualification is:-Associate Degree / Higher Diploma/Diploma',
        3: 'You highest academic qualification is:-Pre-University Programs: STPM/A-Level or equivalent qualification',
        4: 'You highest academic qualification is:-SPM/O-Level/IGCSE or equivalent qualification',
        5: 'You highest academic qualification is:-Others (e.g. technical institute/below high school graduation)',
        6: 'You academic focus includes:-Business/Finance',
        7: 'You academic focus includes:-Information Technology/Data Science/AI',
        8: 'You academic focus includes:-Hospitality and Tourism/Culinary Arts',
        9: 'You academic focus includes:-Arts/History',
        10: 'You academic focus includes:-Mass Communication',
        11: 'You academic focus includes:-Science',
        12: 'You academic focus includes:-Law',
        13: 'You academic focus includes:-Education',
        14: 'You academic focus includes:-Design/Architecture',
        15: 'You academic focus includes:-Engineering',
        16: 'You academic focus includes:-Medicine/Pharmacy',
        17: 'You academic focus includes:-Others',
        18: 'Your most recent occupations include:-Insurance/Takaful',
        19: 'Your most recent occupations include:-Account/Banking/Finance',
        20: 'Your most recent occupations include:-Administrative/Human Resource/Secretariat',
        21: 'Your most recent occupations include:-Consulting/Management',
        22: 'Your most recent occupations include:-Information Technology',
        23: 'Your most recent occupations include:-Executive / Professional',
        24: 'Your most recent occupations include:-Sales/Marketing/Social Media/Writer',
        25: 'Your most recent occupations include:-Self-employed/Business Owner',
        26: 'Your most recent occupations include:-Community Services / Volunteer',
        27: 'How long was your most recent job?-No full-time experience (e.g. incoming graduate)',
        28: 'How long was your most recent job?-less than 6 months',
        29: 'How long was your most recent job?-6 months - less than 1 year',
        30: 'How long was your most recent job?-1 year - less than 3 years',
        31: 'How long was your most recent job?-3 years - less than 5 years ',
        32: 'How long was your most recent job?-5 years or longer',
        33: 'How long was your longest full time job?-No full-time experience (e.g. incoming graduate)',
        34: 'How long was your longest full time job?-Less than 1 year',
        35: 'How long was your longest full time job?-1 year - less than 3 years',
        36: 'How long was your longest full time job?-3 years - less than 5 years',
        37: 'How long was your longest full time job?-5 years - less than 10 years',
        38: 'How long was your longest full time job?-10 years or longer',
        39: 'Your personal income (excl. rental, and interest/dividends from insurance and investment) in the past year is:-Less than RM 30,000',
        40: 'Your personal income (excl. rental, and interest/dividends from insurance and investment) in the past year is:-RM 30,000-49,999',
        41: 'Your personal income (excl. rental, and interest/dividends from insurance and investment) in the past year is:-RM 50,000-99,99',
        42: 'Your personal income (excl. rental, and interest/dividends from insurance and investment) in the past year is:-RM 100,000-199,999',
        43: 'Your personal income (excl. rental, and interest/dividends from insurance and investment) in the past year is:-RM 200,000-299,999',
        44: 'Your personal income (excl. rental, and interest/dividends from insurance and investment) in the past year is:-RM 300,000 or above',
        45: 'Your average family spending (incl. mortgage) per month:-Less than RM 3,000',
        46: 'Your average family spending (incl. mortgage) per month:-RM 3,000-4,999',
        47: 'Your average family spending (incl. mortgage) per month:-RM 5,000-9,999',
        48: 'Your average family spending (incl. mortgage) per month:-RM 10,000-14,999',
        49: 'Your average family spending (incl. mortgage) per month:-RM 15,000-19,999',
        50: 'Your average family spending (incl. mortgage) per month:-RM 20,000 or above',
        51: 'In terms of wealth management, you asset distribution includes:-Cash and Bank Accounts (Savings, FDs)',
        52: 'In terms of wealth management, you asset distribution includes:-Properties / Land',
        53: 'In terms of wealth management, you asset distribution includes:-Employees Provident Fund (EPF) / Private Retirement Scheme (PRS)',
        54: 'In terms of wealth management, you asset distribution includes:-Stocks/Funds',
        55: 'In terms of wealth management, you asset distribution includes:-Others',
        56: 'NA',
        57: 'NA',
        58: 'NA',
        59: 'NA',
        60: 'Your residential situation is:-Live with parents',
        61: 'Your residential situation is:-Self-owned property (subject to mortgage)',
        62: 'Your residential situation is:-Self-owned property (no mortgage)',
        63: 'Your residential situation is:-Private rental residence',
        64: 'Your residential situation is:-Public housing',
        65: 'Your residential situation is:-Other',
        66: 'You have resided in Malaysia for:-Less than 1 year',
        67: 'You have resided in Malaysia for:-1 year - less than 3 years',
        68: 'You have resided in Malaysia for:-3 years - less than 5 years',
        69: 'You have resided in Malaysia for:-5 years - less than 7 years',
        70: 'You have resided in Malaysia for:-7 years or above',
        71: 'Have you purchased life insurance for yourself or your family before? Do you have any claims experience?-Yes, and I have claims experience',
        72: 'Have you purchased life insurance for yourself or your family before? Do you have any claims experience?-Yes, but I do not have any claims experience',
        73: 'Have you purchased life insurance for yourself or your family before? Do you have any claims experience?-No',
        74: 'Do you have any relatives or friends in the insurance industry?-Yes, and they are close with me',
        75: 'Do you have any relatives or friends in the insurance industry?-Yes, but I am not close with them',
        76: 'Do you have any relatives or friends in the insurance industry?-No',
        88: 'How many contacts are there in your phone book and social media?-Fewer than 100',
        89: 'How many contacts are there in your phone book and social media?-100 - 299',
        90: 'How many contacts are there in your phone book and social media?-300 - 499',
        91: 'How many contacts are there in your phone book and social media?-500 - 999',
        92: 'How many contacts are there in your phone book and social media?-1000 - 4999',
        93: 'How many contacts are there in your phone book and social media?-5000 or more',
        94: 'Your contacts are primarily made up of:-Schoolmates',
        95: 'Your contacts are primarily made up of:-Colleagues',
        96: 'Your contacts are primarily made up of:-Relatives',
        97: 'Your contacts are primarily made up of:-Salespersons',
        98: 'Your contacts are primarily made up of:-Business Partners / Clients',
        99: 'How often do you update your social media?-Daily',
        100: 'How often do you update your social media?-Weekly',
        101: 'How often do you update your social media?-Monthly',
        102: 'How often do you update your social media?-Never',
        103: 'Social circle located in the same city',
        104: 'Social circle located in the same state/province.',
        105: 'Social circle located in other states in Malaysia.',
        106: 'Social circle located outside of Malaysia',
        107: 'NA',
        108: 'NA',
        109: 'Your ideal career should bear which of the following important elements?-Independence',
        110: 'Your ideal career should bear which of the following important elements?-Recognition',
        111: 'Your ideal career should bear which of the following important elements?-Power',
        112: 'Your ideal career should bear which of the following important elements?-Self-esteem',
        113: 'Your ideal career should bear which of the following important elements?-Monetary Return',
        114: 'Your ideal career should bear which of the following important elements?-Family Life',
        115: 'Your ideal career should bear which of the following important elements?-Personal Time',
        116: 'Your ideal career should bear which of the following important elements?-Reputation',
        117: 'Your ideal career should bear which of the following important elements?-Personal Development',
        118: 'Your ideal career should bear which of the following important elements?-Challenge',
        119: 'Your ideal career should bear which of the following important elements?-Achievement ',
        120: 'Your ideal career should bear which of the following important elements?-Sense of Security',
        121: 'How did you learn about this role?-AIA Website',
        122: 'How did you learn about this role?-AIA Career Orientation Program (COP)',
        123: 'How did you learn about this role?-Career Expo',
        124: 'How did you learn about this role?-Referred by Friends / Relatives',
        125: 'How did you learn about this role?-Referred by Existing AIA Agent / Leader',
        126: 'How did you learn about this role?-Advertisement: Newspaper / Internet / Others',
        127: 'How did you learn about this role?-Educational Institution',
        128: 'How did you learn about this role?-Others',
        129: 'Which age group do you belong to?-Below 20',
        130: 'Which age group do you belong to?-20-24 years old',
        131: 'Which age group do you belong to?-25-29 years old ',
        132: 'Which age group do you belong to?-30-34 years old',
        133: 'Which age group do you belong to?-35-39 years old',
        134: 'Which age group do you belong to?-40-44 years old',
        135: 'Which age group do you belong to?-45-49 years old',
        136: 'Which age group do you belong to?-50-54 years old',
        137: 'Which age group do you belong to?-55-59 years old',
        138: 'Which age group do you belong to?-60-64 years old',
        139: 'Which age group do you belong to?-65 or above',
        140: 'Which age group do you belong to?-Prefer not to answer',
        141: 'How do you think one should strike a balance between work and life?-In order to maintain family and personal needs, career development should come first',
        142: 'How do you think one should strike a balance between work and life?-Be properly rested before starting to work, so as to go further',
        143: 'How do you think one should strike a balance between work and life?-Work and life are complementary, there is no need to draw a clear line',
        144: 'A meeting with an important client is to be held in 5 days. It has just been rescheduled to tomorrow but you have yet to prepare for it, you would:-Halt all other work and start preparing for the client meeting',
        145: 'A meeting with an important client is to be held in 5 days. It has just been rescheduled to tomorrow but you have yet to prepare for it, you would:-Rearrange the priority of your work, and make time for preparation',
        146: 'A meeting with an important client is to be held in 5 days. It has just been rescheduled to tomorrow but you have yet to prepare for it, you would:-First finish the arranged tasks, try my best to prepare for the meeting in the remaining time',
        147: 'Your friend just opened a new company and asks you to introduce your own clients, you would:-Not share any client information, but only introduce the friend`s company to my clients who are interested',
        148: 'Your friend just opened a new company and asks you to introduce your own clients, you would:-Share clients preferences with my friend for his reference',
        149: 'Your friend just opened a new company and asks you to introduce your own clients, you would:-Only share my experience of client relationship management and how to extend the client base',
        150: 'After your trip, you found out that you have overspent the planned budget, you would:-Always strictly stay within budget and never overspend',
        151: 'After your trip, you found out that you have overspent the planned budget, you would:-Save money in the coming days and make a plan to avoid overspending next time',
        152: 'After your trip, you found out that you have overspent the planned budget, you would:-It is not a problem at all, I am just treating myself a bit',
        153: 'You discovered that a colleague has erred in a task, yet he makes excuses and refuses to rectify, you would:-Work together to solve the problem for the team first',
        154: 'You discovered that a colleague has erred in a task, yet he makes excuses and refuses to rectify, you would:-Report to the manager and ask for advice',
        155: 'You discovered that a colleague has erred in a task, yet he makes excuses and refuses to rectify, you would:-I have already tried my best to remind him, there is nothing else I can do if he does not listen to me',
        156: 'You previously offered special discount to a client, yet he did not contact you until you have left the job, you would:-Contact the previous company myself to fight for the offer, to realise what I promised',
        157: 'You previously offered special discount to a client, yet he did not contact you until you have left the job, you would:-Ask my previous colleague to follow up and contact the client',
        158: 'You previously offered special discount to a client, yet he did not contact you until you have left the job, you would:-Inform the client that you have already resigned and offer him a meal to show that you are sorry',
        159: 'When was the last time that you learned?-I spare time to learn every week regularly ',
        160: 'When was the last time that you learned?-I learn from my colleagues when I encounter difficulties',
        161: 'When was the last time that you learned?-I learn by mandatory courses offered by company ',
        162: 'After a meeting, you feel confused about one of the decisions, you would:-Try to understand the decision, believing that I will know the reason in time',
        163: 'After a meeting, you feel confused about one of the decisions, you would:-Ask my colleagues after the meeting until I clearly understand',
        164: 'After a meeting, you feel confused about one of the decisions, you would:-The decision has already been made, there is no need to go deep into it',
        165: 'Which of the following are you more interested in?-Deep dive to learn the topics that I am interested in',
        166: 'Which of the following are you more interested in?-Understand myself and the people around me',
        167: 'Which of the following are you more interested in?-Explore new places, meet new friends and have new experience',
        168: 'A box-office hit is screening in town, yet it is not a genre that you typically like, you would:-Go watch the movie so I can join the discussion among friends',
        169: 'A box-office hit is screening in town, yet it is not a genre that you typically like, you would:-Read the movie reviews before deciding to watch it or not',
        170: 'A box-office hit is screening in town, yet it is not a genre that you typically like, you would:-It is not a genre I like so I am not interested in watching',
        171: 'Your manager invited you to work on a task that you are not familiar with, you would:-Thank for the opportunity and say yes immediately',
        172: 'Your manager invited you to work on a task that you are not familiar with, you would:-Ask for more details about the task before deciding',
        173: 'Your manager invited you to work on a task that you are not familiar with, you would:-Learn the related knowledge before taking up the challenge',
        174: 'Where do you expect yourself to be in 5 years:-Provide excellent services to clients and have a good reputation',
        175: 'Where do you expect yourself to be in 5 years:-Become one of the top 10 Financial Planners in the company ',
        176: 'Where do you expect yourself to be in 5 years:-Have my own team of 100 members',
        177: 'Which of the following best represents your life attitude?-Go with the flow - enjoy the journey',
        178: 'Which of the following best represents your life attitude?-Swim upstream - try to realise my goals enthusiastically',
        179: 'Which of the following best represents your life attitude?-Live my days and enjoy my life - resting is as essential as setting and realising goals',
        180: 'Do you wish to be recognised by others?-I would love to be recognised by everyone',
        181: 'Do you wish to be recognised by others?-I would be happy as long as I get recognition from my close circle',
        182: 'Do you wish to be recognised by others?-I am satisfied with my ways of life and do not care much about how others view me',
        183: 'When someone compliments you, you would:-Take it, compliment is one of the greatest motivations for me to be successful',
        184: 'When someone compliments you, you would:-Feel puzzled, doubt if he really means it',
        185: 'When someone compliments you, you would:-Thank them and humbly say that it is just luck',
        186: 'What is your way of working?-Be prudent and careful, try my best to make precautions and follow-up actions',
        187: 'What is your way of working?-Keep my eye on the goals, work my best to achieve them',
        188: 'What is your way of working?-Step by step, work according to the existing mechanism and guidance from team',
        189: 'On social media, you saw that your friend is planning to travel to a place that you have recently visited, you would:-Call my friend to recommend some must-visit spots',
        190: 'On social media, you saw that your friend is planning to travel to a place that you have recently visited, you would:-Ask my friend out to recommend itinerary, offer to help contact local driver and B&B',
        191: 'On social media, you saw that your friend is planning to travel to a place that you have recently visited, you would:-Comment on the post, and offer to help out anytime',
        192: 'During festive season, you usually would:-Draft specific messages in advance and send to close friends and family',
        193: 'During festive season, you usually would:-Draft a message in advance and send to every friend and group',
        194: 'During festive season, you usually would:-Reply thank you after receiving messages from others',
        195: 'When you sleep in an unfamiliar bed during travelling, you would:-Have no insomnia, not feel much difference from staying at home',
        196: 'When you sleep in an unfamiliar bed during travelling, you would:-Have insomnia sometimes',
        197: 'When you sleep in an unfamiliar bed during travelling, you would:-Have real bad insomnia even if I change my posture and pillow',
        198: 'When you just joined a new team, you would:-Take the initiative to ask colleagues to know more about the job',
        199: 'When you just joined a new team, you would:-Grab lunch with the colleagues to blend into the team',
        200: 'When you just joined a new team, you would:-Get to know each other gradually through work',
        201: 'At the critical moment determining success or failure, you would:-Feel a little nervous, but calm down very quickly and get into my best state',
        202: 'At the critical moment determining success or failure, you would:-Fully focus, pay attention to details to avoid mistakes from anxiety',
        203: 'At the critical moment determining success or failure, you would:-Keep a calm mind, knowing that even that I lose this time, there is always a time for winning',
        204: 'You have planned to go to your friend`s place to have dinner, but he has to take a rain check because of some urgency, you would:-Change my plan, go to a popular coffee shop nearby instead',
        205: 'You have planned to go to your friend`s place to have dinner, but he has to take a rain check because of some urgency, you would:-Feel disappointed and go home',
        206: 'You have planned to go to your friend`s place to have dinner, but he has to take a rain check because of some urgency, you would:-Contact my friend to meet up again next time',
        207: 'Your company is holding a grand annual party, you would:-Blend into different groups and actively talk to others',
        208: 'Your company is holding a grand annual party, you would:-Get to know new friends through introduction from colleagues',
        209: 'Your company is holding a grand annual party, you would:-Take the chance to hang out with colleagues who are close with you',
        210: 'If you suddenly become the focus in a gathering, you would:-Lead the ambience there and make the gathering more fun',
        211: 'If you suddenly become the focus in a gathering, you would:-Enjoy being centre of attention',
        212: 'If you suddenly become the focus in a gathering, you would:-Just go with the flow and avoid catching too much attention',
        213: 'When you are in disagreement with colleagues at work, you would:-Seek common ground and try to convince others to stand with me ',
        214: 'When you are in disagreement with colleagues at work, you would:-Listen to others humbly, consider others views',
        215: 'When you are in disagreement with colleagues at work, you would:-Consult your manager or other colleagues advice',
        216: 'Which of the following descriptions best suits you:-I want to be recognised by others, gather the team to achieve the same goal',
        217: 'Which of the following descriptions best suits you:-I can win others trust with my own ability and judgement ',
        218: 'Which of the following descriptions best suits you:-I will carefully think about others feelings to avoid confrontation',
        219: 'What is your usual role in a gathering with friends:-Organiser, actively arrange the event',
        220: 'What is your usual role in a gathering with friends:-Coordinator, contact your close friends to join',
        221: 'What is your usual role in a gathering with friends:-Participant, enjoy joining team activities',
        222: 'A member in your team failed to reach his sales target for the past 3 months, you would:-Encourage him and recognise his efforts',
        223: 'A member in your team failed to reach his sales target for the past 3 months, you would:-Invite him to have lunch together and ask him the reason behind ',
        224: 'A member in your team failed to reach his sales target for the past 3 months, you would:-Set up an improvement plan for him, and follow up with him myself',
        225: 'If you cannot sign any deal after joining the company for 2 months, you would:-Understand my own problems and try to solve them in different dimensions',
        226: 'If you cannot sign any deal after joining the company for 2 months, you would:-Ask my manager about how can you improve',
        227: 'If you cannot sign any deal after joining the company for 2 months, you would:-Keep trying, give myself another chance',
        228: 'If your client backs off right before signing contract, you would:-Keep going, try to convince and move the client',
        229: 'If your client backs off right before signing contract, you would:-Understand the reason of failure, keep reaching out to the next client',
        230: 'If your client backs off right before signing contract, you would:-Give up on this client, maybe he is not for me',
        231: 'When you face criticism or accusation, you would:-Accept the criticism humbly, and welcome any other feedbacks in the future',
        232: 'When you face criticism or accusation, you would:-Ask for others opinions, confirm that everyone shares the same view and makes improvements',
        233: 'When you face criticism or accusation, you would:-Feel very disappointed, think that the criticiser cannot understand me, and talk to colleagues about it',
        234: 'If your progress is falling behind from the target you set, you would:-Work hard to make up the progress, not be affected by my emotions',
        235: 'If your progress is falling behind from the target you set, you would:-Try again in another way',
        236: 'If your progress is falling behind from the target you set, you would:-Shift my focus to other goals first, try again when I am in a better state',
        237: 'You are asked to take over a difficult task while you are expected to perform excellently, you would:-I am confident about my ability, I can go into the state easily',
        238: 'You are asked to take over a difficult task while you are expected to perform excellently, you would:-Feel stressed, but try my best to finish the task',
        239: 'You are asked to take over a difficult task while you are expected to perform excellently, you would:-Go with the flow, avoid letting stress to affect my work',
        240: 'Your peer gifts his friends policies and even pays the premiums on their behalf in order to make extra bonus. You would:-Follow the example of your peer when you encounter a similar situation',
        241: 'Your peer gifts his friends policies and even pays the premiums on their behalf in order to make extra bonus. You would:-Do nothing, because this is a win-win situation',
        242: 'Your peer gifts his friends policies and even pays the premiums on their behalf in order to make extra bonus. You would:-Report to my manager because there should be something not quite right',
        243: 'In order to support your career development, your client gives you a number of business cards of his business partners for you to conduct marketing activities with them. You would:-Thank for the clientï¿½s support and actively conduct marketing activities with these persons',
        244: 'In order to support your career development, your client gives you a number of business cards of his business partners for you to conduct marketing activities with them. You would:-Express your gratitude and further ask the client to obtain proper consents from the business partners before accepting the business cards',
        245: 'In order to support your career development, your client gives you a number of business cards of his business partners for you to conduct marketing activities with them. You would:-Keep them up for future use',
        246: 'A client discloses to you that he had suffered from certain illness, but requests you not to state it in the insurance application forms. You would:-Accede to the clientï¿½s request and not disclose the illness in the application forms',
        247: 'A client discloses to you that he had suffered from certain illness, but requests you not to state it in the insurance application forms. You would:-Understand more about the illness and assess if it would affect the underwriting decision based on your personal knowledge',
        248: 'A client discloses to you that he had suffered from certain illness, but requests you not to state it in the insurance application forms. You would:-Explain to the client that full disclosure of medical history is needed for the Company to make correct underwriting decision'}}

    # Selected features for model input

    selected_cols = [
        'Q001_Self-conscientiousness',
        'Q002_Self-conscientiousness',
        'Q003_Self-conscientiousness',
        'Q005_Self-conscientiousness',
        'Q006_Self-conscientiousness',
        'Q072_Self-conscientiousness',
        'Q007_Curiosity',
        'Q010_Curiosity',
        'Q011_Curiosity',
        'Q012_Curiosity',
        'Q013_Curiosity',
        'Q014_Ambition',
        'Q015_Ambition',
        'Q018_Ambition',
        'Q019_Ambition',
        'Q020_Ambition',
        'Q021_Proactiveness',
        'Q022_Proactiveness',
        'Q031_Proactiveness',
        'Q032_Proactiveness',
        'Q025_Agility',
        'Q026_Agility',
        'Q028_Agility',
        'Q029_Agility',
        'Q034_Leadership',
        'Q035_Leadership',
        'Q036_Leadership',
        'Q038_Leadership',
        'Q039_Resilience',
        'Q040_Resilience',
        'Q041_Resilience',
        'Q042_Resilience',
        'Q043_Resilience',
        'Q061_Network Resources',
        'Q062_Network Resources',
        'Q064_Network Resources',
        'Q073_Compliance Awareness',
        'Q074_Compliance Awareness',
        'Q075_Compliance Awareness',
        
        'Q052_BackGround_Count',
        'Q060_BackGround_Count',
        'Q056_BackGround_Count',
        'Expend_Over_Income',
        'Q054_A',
        'Q054_B',
        'Q054_C',
        'Q054_D',
        'Q054_E',
        'Q055_A',
        'Q055_B',
        'Q055_C',
        'Q055_D',
        'Q055_E',
        'Q055_F',
        'Q057_A',
        'Q057_B',
        'Q057_C',
        'Q057_D',
        'Q057_E',
        'Q057_F',
        'Q058_A',
        'Q058_B',
        'Q058_C',
        'Q058_D',
        'Q058_E',
        'Q058_F',
        'Q067_A',
        'Q067_B',
        'Q067_C',
        'Q076_A',
        'Q076_B',
        'Q076_C',
        'Q076_D',
        'Q076_E',
        'Q076_F',
        'Q076_G',
        'Q076_H',
        'Q076_I',
        'Q076_J',
        'Q076_K',
        'Q076_L',
        'Agility',
        'Ambition',
        'Compliance Awareness',
        'Curiosity',
        'Leadership',
        'Network Resources',
        'Proactiveness',
        'Resilience',
        'Self-conscientiousness',
        'Cognitive ability'
        ]
    approach = 3
    columns = selected_cols
    string = input_json

    # Convert Json file into Dataframe
    df = pd.json_normalize(string, sep=',')
    df = df.explode('categorys')

    col = 'categorys'
    nested_df = pd.json_normalize(df[col])
    nested_df.columns = [sub_col for sub_col in nested_df.columns]
    df0 = pd.merge(df.reset_index(), nested_df.reset_index(), left_index=True, right_index=True)
    df0 = df0.drop(columns = [col,'index_x','index_y'])
    df1 = df0.explode('questionItems')

    col = 'questionItems'
    nested_df = pd.json_normalize(df1[col])
    nested_df.columns = [sub_col for sub_col in nested_df.columns]
    df2 = pd.merge(df1.reset_index(), nested_df.reset_index(), left_index=True, right_index=True)
    df2 = df2.drop(columns = [col,'index_x','index_y'])
    df3 = df2.explode('answerItems')

    col = 'answerItems'
    nested_df = pd.json_normalize(df3[col])
    nested_df.columns = [sub_col for sub_col in nested_df.columns]
    df4 = pd.merge(df3.reset_index(), nested_df.reset_index(), left_index=True, right_index=True)
    df4 = df4.drop(columns = [col,'index_x','index_y'])
    df4['agent_cd'] = df4['id']

    # Convert mapping json into dataframe
    full = pd.DataFrame(full)
    full['questionCode'] = full['Question_AnswerCode'].str.split('_').str[0]
    full['question'] = full['Question_code'].str.split('_').str[0]

    # Get max value of each competency
    compentency_maxScore = {'categoryCode':{1:'Self-conscientiousness',
    2:'Curiosity',
    3:'Ambition',
    4:'Sociability',
    5:'Adaptability',
    6:'Compliance Awareness',
    7:'Network Resources',
    8:'Leadership',
    9:'Resilience',
    10:'Cognitive ability'},
    'max_score':{1:12,
    2:10,
    3:10,
    4:8,
    5:8,
    6:6,
    7:7.5,
    8:8,
    9:10,
    10:10}}

    # Standardize Compentency 
    compentency_maxScore = pd.DataFrame(compentency_maxScore)
    compentency = df4[['agent_cd','categoryCode','categoryScore']].drop_duplicates()
    compentency = pd.merge(compentency_maxScore,compentency, on = 'categoryCode', how ='left')
    compentency['scaled_score'] = compentency['categoryScore']/compentency['max_score']
    compentency = compentency.pivot_table(index = 'agent_cd', columns='categoryCode', values='scaled_score',aggfunc='first').reset_index()


    # Get Compentenecy Existence Count
    question_layer = df4[df4['categoryCode']!='Others'][['agent_cd','questionCode','answerCode','score']].drop_duplicates()
    question_layer = question_layer.rename(columns={"answerCode": "Question_AnswerCode_s"})
    question_layer['temp_question_layer'] = 1
    question_layer_temp = pd.merge(question_layer[['agent_cd','questionCode','Question_AnswerCode_s','temp_question_layer','score']].drop_duplicates(),full[full['Category']!='Others'][['questionCode','Question_AnswerCode']].drop_duplicates(), on = "questionCode", how = 'right')

    question_layer = question_layer.rename(columns={"Question_AnswerCode_s": "Question_AnswerCode"})
    question_layer = question_layer.rename(columns={"temp_question_layer": "temp_answer_layer"})
    answer_layer_temp = pd.merge(question_layer_temp, question_layer[['agent_cd','Question_AnswerCode',"temp_answer_layer"]].drop_duplicates(), on = ['agent_cd','Question_AnswerCode'], how = 'left')

    answer_layer_temp['Exist'] = np.where((answer_layer_temp['temp_question_layer'] == 1) & (answer_layer_temp['temp_answer_layer'] == 1), 1, np.where((answer_layer_temp['temp_question_layer'] == 1) & (answer_layer_temp['temp_answer_layer'] != 1), 0,-1))
    approach_question_set = pd.merge(full[['Question_code','Question_AnswerCode','question','Category']].drop_duplicates(), answer_layer_temp.drop_duplicates(), on = 'Question_AnswerCode', how = 'inner')

    approach_question_set['Q_id'] = approach_question_set['question'] + '_' + approach_question_set['Category']
    approach_question_set['score2'] = approach_question_set['score']*10
    approach_question_set['score2'] = approach_question_set['score2'].fillna(0)
    approach_question_set['score2'] = approach_question_set['score2'].astype('int').astype('str')
    approach_question_set['Q_id2'] = approach_question_set['question'] + '_Score_' + approach_question_set['score2']
    approach_question_set['C_id2'] = approach_question_set['Category'] + '_Score_' + approach_question_set['score2']

    approach_question_set['score'] = approach_question_set['score']/2
    
    approach_1_questions = approach_question_set.pivot(index = 'agent_cd', columns='Question_code', values='Exist')
    approach_1_questions = approach_1_questions.reset_index()
    approach_3_questions = approach_question_set[['agent_cd','Q_id','score',]].drop_duplicates().fillna(-1).pivot(index = 'agent_cd', columns='Q_id', values='score').reset_index()

    
    # Get Background INFO
    background = df4[df4['categoryCode']=='Others'][['agent_cd','questionCode','answerCode','text']].drop_duplicates()
    background['temp_question_layer'] = 1
    background = background.rename(columns={"answerCode": "Question_AnswerCode_s"})

    background_temp = pd.merge(background[['agent_cd','questionCode','Question_AnswerCode_s','temp_question_layer']].drop_duplicates(),full[full['Category']=='Others'][['Question_code','questionCode','Question_AnswerCode']].drop_duplicates(), on = "questionCode", how = 'right')

    background = background.rename(columns={"Question_AnswerCode_s": "Question_AnswerCode"})
    background = background.rename(columns={"temp_question_layer": "temp_answer_layer"})

    background_temp = pd.merge(background_temp, background[['agent_cd','Question_AnswerCode',"temp_answer_layer"]].drop_duplicates(), on = ['agent_cd','Question_AnswerCode'], how = 'left')

    background_temp['Exist'] = np.where((background_temp['temp_question_layer'] == 1) & (background_temp['temp_answer_layer'] == 1), 1, np.where((background_temp['temp_question_layer'] == 1) & (background_temp['temp_answer_layer'] != 1), 0,-1))

    # Calculate Expand over Income
    conditions = [
    background_temp['Question_code'] == 'Q050_A',
    background_temp['Question_code'] == 'Q050_B',
    background_temp['Question_code'] == 'Q050_C',
    background_temp['Question_code'] == 'Q050_D',
    background_temp['Question_code'] == 'Q050_E',
    background_temp['Question_code'] == 'Q050_F'
    ]
    #values = [100000, 150000,250000,400000,750000,1000000]
    values = [3000,5000,10000,15000,19000,20000]

    background_temp['q050_score'] = np.select(conditions,values)

    conditions = [
        background_temp['Question_code'] == 'Q051_A',
        background_temp['Question_code'] == 'Q051_B',
        background_temp['Question_code'] == 'Q051_C',
        background_temp['Question_code'] == 'Q051_D',
        background_temp['Question_code'] == 'Q051_E',
        background_temp['Question_code'] == 'Q051_F'
    ]
    values = [3000,5000,10000,15000,19000,20000]

    background_temp['q051_score'] = np.select(conditions,values)

    background_temp_expand = background_temp[(background_temp['Exist'] == 1)&((background_temp['q051_score'] !=0)|(background_temp['q050_score'] !=0))]
    background_temp_expand = background_temp_expand.groupby('agent_cd').agg({'q050_score':'max','q051_score':'max'}).reset_index()
    background_temp_expand['Expend_Over_Income'] = background_temp_expand['q051_score'] / background_temp_expand['q050_score']
    
    # Count of BackGround Question Answers
    background = background_temp[['agent_cd','Question_code','Exist']].drop_duplicates()
    background = background.pivot(index = 'agent_cd', columns='Question_code', values='Exist')
    background = background.merge(background_temp_expand[['agent_cd','Expend_Over_Income']], on = 'agent_cd', how ='inner')

    background['Q056_BackGround_Count'] = background['Q056_A']+background['Q056_B']+background['Q056_C']+background['Q056_D']+background['Q056_E']+background['Q056_F']+background['Q056_G']+background['Q056_H']+background['Q056_I']+background['Q056_J']+background['Q056_K']+background['Q056_L']
    background['Q052_BackGround_Count'] = background['Q052_A']+background['Q052_B']+background['Q052_C']+background['Q052_D']+background['Q052_E']+background['Q052_F']+background['Q052_G']+background['Q052_H']+background['Q052_I']
    background['Q060_BackGround_Count'] = background['Q060_A']+background['Q060_B']+background['Q060_C']+background['Q060_D']+background['Q060_E']+background['Q060_F']+background['Q060_G']+background['Q060_H']+background['Q060_I']

    input_data = pd.merge(compentency, approach_3_questions, on ='agent_cd', how = 'inner')
    input_data = pd.merge(input_data, background, on ='agent_cd', how = 'inner')


    # Rename column name to match ouput list
    for col in input_data.columns:
        if col not in columns:
            if col == 'agent_cd':
                continue
            elif 'Adaptability' in col:
                new_col = col.replace('Adaptability','Agility')
                input_data.rename(columns={col: new_col},inplace=True)
            elif 'Sociability' in col:
                new_col = col.replace('Sociability','Proactiveness')
                input_data.rename(columns={col:new_col},inplace=True)
            else:
                input_data[col] = -1


    input_data = input_data[columns+['agent_cd']]
    #print(list(columns))
    # Load model 
    columns = ['Q001_Self-conscientiousness', 'Q002_Self-conscientiousness', 'Q003_Self-conscientiousness', 
        'Q005_Self-conscientiousness', 'Q006_Self-conscientiousness', 'Q072_Self-conscientiousness', 
        'Q007_Curiosity', 'Q011_Curiosity', 'Q012_Curiosity', 'Q013_Curiosity', 
        'Q014_Ambition', 'Q015_Ambition', 'Q018_Ambition', 'Q019_Ambition', 'Q020_Ambition', 'Q021_Proactiveness', 
        'Q022_Proactiveness', 'Q031_Proactiveness', 'Q032_Proactiveness', 'Q025_Agility', 'Q026_Agility', 'Q028_Agility', 
        'Q029_Agility', 'Q034_Leadership', 'Q035_Leadership', 'Q036_Leadership', 'Q038_Leadership', 'Q039_Resilience', 
        'Q040_Resilience', 'Q041_Resilience', 'Q042_Resilience', 'Q043_Resilience', 'Q061_Network Resources', 
        'Q062_Network Resources', 'Q064_Network Resources', 'Q073_Compliance Awareness', 'Q074_Compliance Awareness',
        'Q075_Compliance Awareness', 'Expend_Over_Income',
        'Agility', 'Ambition', 'Compliance Awareness', 'Curiosity', 'Leadership', 'Network Resources', 'Proactiveness', 
        'Resilience', 'Self-conscientiousness','Cognitive ability']
   
    model_data = input_data[columns]
    #print(model_data)
    
    # Prediction
    # Predict probability of each agent
    if model is None:
        print("Error: Model is None. Check loading process.")
    else:
        #probabilities = model.predict_proba(model_data)[:, 1]
        probabilities = model.predict(model_data)
        print('prediction finished')

    # # Shap value to get top important features
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(model_data)
    if len(shap_values) > 1:
      shap_values = shap_values[1]
      

    responses = []

    # # get key competency and growth for each record
    s = 0
    for agent in input_data['agent_cd'].to_list():
        df_shap = input_data[input_data['agent_cd'] == agent][columns].iloc[0].to_frame()
        df_shap = df_shap.reset_index(inplace=False)
        df_shap.columns = ['feature','value']
        df_shap['shap'] = shap_values[s,:]
        df_shap = df_shap[df_shap['feature'].isin(['Ambition','Curiosity','Leadership','Network Resources','Resilience','Self-conscientiousness','Agility','Proactiveness','Cognitive ability','Compliance Awareness'])]

        key_competency = 'NO_COMPETENCY_FOUND'
        df_shap = df_shap.sort_values(by = 'shap', ascending = False)
        #df_shap.display()
        for index, row in df_shap.iterrows():
            if row['value'] > 0 and row['shap'] > 0:
                key_competency = row['feature']
                break

        # # If no positive contributers
        # if df_shap.loc[df_shap['shap'].idxmax()]['shap'] < 0:
        #     key_competency = 'NO_COMPETENCY_FOUND'

        columns_f = ['feature', 'marginal_increase']
        out = pd.DataFrame(columns=columns_f)

        input_temp = input_data[input_data['agent_cd'] == agent][columns]

    #     # Get key growth of each record by rank most marginal increase for each feature
        for col in ['Ambition','Compliance Awareness','Curiosity','Leadership','Network Resources','Resilience','Self-conscientiousness','Agility','Proactiveness','Cognitive ability','Compliance Awareness']:
            maxs = mins = probabilities[s]
            max_diff = 0
            for i in range(0,11):
                if col in input_temp.columns:
                    input_temp[col] = i/10
                    #probabilities_temp = model.predict(input_temp)#[:, 1][0]
                    probabilities_temp = model.predict(input_temp)[0]
                    if probabilities_temp > maxs:
                        maxs = probabilities_temp
                    if probabilities_temp < mins:
                        mins = probabilities_temp
                else:
                    continue

            input_temp[col] = input_data[col].values[0]
            max_diff = maxs - mins
            out.loc[len(out)] = [col, max_diff]

        # Use the feature that contriute most increase in performance
        key_growth = out.loc[out['marginal_increase'].idxmax()]['feature']

        if key_growth == key_competency:
            out = out[out['feature']!= key_growth]
            key_growth = out.loc[out['marginal_increase'].idxmax()]['feature']

        # Convert output name to the given standard
        if key_competency == 'Agility':
            key_competency = 'Adaptability' 
        elif key_competency == 'Proactiveness':
            key_competency = 'Sociability'
        elif key_competency == 'Cognitive ability':
            key_competency = 'Cognitive Ability'

        if key_growth == 'Agility':
            key_growth = 'Adaptability'
        elif key_growth == 'Proactiveness':
            key_growth = 'Sociability'
        elif key_growth == 'Cognitive ability':
            key_growth = 'Cognitive Ability'
        
        
        # Setup Output Structure
        response = {
            'id':agent,
            'score': probabilities[s],
            'keyGrowth': key_growth,
            'strongest': key_competency
        }
        
        responses.append(response)
        s+=1

    # # print(responses)

    #return probabilities,model_data,shap_values,responses
    return responses

# response wrapper
class R(object):

    @staticmethod
    def ok(payload):
        result = {"code": 200, "message": "SUCCESS", "payload": payload}
        return result

    @staticmethod
    def fail(code, msg, errors):
        result = {"code": code, "message": msg, "errors": errors}
        return result


app = FastAPI(title="my-datasciencelab-app-agency-cat", version="1.0.0")


# @app.on_event("startup")
# def startup_event():
#     init()


@app.get("/health")
def healthcheck():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/score")
def score_endpoint(payload: Dict[str, Any]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        result = score(payload)
        return R.ok(result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run("score:app", host="0.0.0.0", port=8000, reload=False)

