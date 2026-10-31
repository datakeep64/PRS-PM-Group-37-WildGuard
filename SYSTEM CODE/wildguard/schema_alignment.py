"""Knowledge model that maps LEMIS and CITES onto one shared schema.

LEMIS (US Fish & Wildlife Service declarations) and the CITES Trade Database describe the
same kind of event - a wildlife product crossing a border - but encode it differently:

  * product form: LEMIS 3-letter codes (LIV, SKI, ...) vs CITES English terms ("live", "skins")
  * units:        LEMIS codes (NO, KG, GM, ...)       vs CITES words ("kg", blank = number)
  * source / purpose: the same letters, except pre-Convention (LEMIS P, CITES O) and
    reintroduction (LEMIS Y, CITES N)
  * countries: ISO-3166 alpha-2 in both, plus a few LEMIS codes CITES never uses
  * taxonomy: class names differ for fish; higher taxa are stored differently

The shared form uses the CITES codes. Product forms and units are mapped to coarse *product
groups* and *unit groups* so that a model trained on LEMIS can be applied to CITES records.
The rules, and the findings behind them, are in DATA/ALIGNING_LEMIS_AND_CITES.md section 2.1.
"""

# --- Product form -------------------------------------------------------------------------
# Coarse groups keep enough distinctions to matter for risk (live vs dead, raw vs worked,
# ivory/horn/bone kept separate) while staying aligned across the two vocabularies. Groups are
# defined by how each vocabulary records an object, not by matching names (notebook 06):
#   - CITES records shoes, and nearly all manufactured leather goods, as "leather products
#     (small)"; LEMIS splits them into small and large goods and shoes. One leather group.
#   - CITES records live rock as raw coral; LEMIS has its own code (ROC).
LEMIS_DESCRIPTION_TO_GROUP = {
    "LIV": "live", "EGL": "live_eggs", "FIG": "live", "ROC": "coral_raw",
    "BOD": "bodies", "DEA": "bodies", "TRO": "trophies",
    "SKI": "skins", "SID": "skins", "PLA": "skins", "RUG": "skins",
    "SKP": "skin_pieces",
    "LPS": "leather_products", "LPL": "leather_products", "SHO": "leather_products",
    "TRI": "leather_products",
    "GAR": "garments", "CLO": "cloth",
    "SPE": "specimens",
    "MEA": "meat", "LEG": "meat", "SOU": "meat", "CAL": "meat",
    "CAV": "caviar",
    "IVC": "ivory", "IVP": "ivory", "IJW": "ivory", "TUS": "ivory", "KEY": "ivory", "PIV": "ivory",
    "HOR": "horn", "HOC": "horn", "HOP": "horn",
    "BON": "bone", "BOP": "bone", "BOC": "bone", "SKE": "bone", "SKU": "skulls",
    "TEE": "teeth_claws", "CLA": "teeth_claws",
    "COR": "coral_raw", "CPR": "coral_raw",
    "SHE": "shells", "SPR": "shells", "CAP": "shells", "SCA": "scales",
    "FEA": "feathers", "WNG": "feathers",
    "EGG": "eggs", "ESH": "eggs",
    "HAI": "hair", "HAP": "hair",
    "FIN": "fins", "SWI": "fins",
    "EXT": "extracts_medicine", "MED": "extracts_medicine", "DER": "extracts_medicine",
    "OIL": "extracts_medicine", "POW": "extracts_medicine", "GAB": "extracts_medicine",
    "GAL": "extracts_medicine", "MUS": "extracts_medicine", "WAX": "extracts_medicine",
    "GEN": "extracts_medicine",
    "CAR": "carvings_jewellery", "JWL": "carvings_jewellery",
    "TAI": "body_parts", "FOO": "body_parts", "EAR": "body_parts", "NES": "body_parts",
    "BAL": "body_parts",
    "UNS": "other",
    # plant, wood and plant-fibre products (VEN is wood veneer, FIB plant fibre)
    "SEE": "other", "WPR": "other", "PLY": "other", "ROO": "other", "FLO": "other",
    "GRS": "other", "SDL": "other", "CUT": "other", "CHP": "other", "FRU": "other",
    "VEN": "other", "SAW": "other", "DPL": "other", "FIB": "other", "TIM": "other",
    "LOG": "other", "CUL": "other", "STE": "other", "BAR": "other", "LVS": "other",
}

CITES_TERM_TO_GROUP = {
    "live": "live", "eggs (live)": "live_eggs", "fingerlings": "live", "pupae": "live",
    "bodies": "bodies", "trophies": "trophies",
    "skins": "skins", "sides": "skins", "plates": "skins", "rug": "skins",
    "skin pieces": "skin_pieces", "skin scraps": "skin_pieces", "leather": "skin_pieces",
    "leather products (small)": "leather_products", "leather items": "leather_products",
    "fur product (small)": "leather_products",
    "leather products (large)": "leather_products", "fur products (large)": "leather_products",
    "shoes": "leather_products",
    "garments": "garments", "cloth": "cloth",
    "specimens": "specimens",
    "meat": "meat", "frog legs": "meat", "soup": "meat", "calipee": "meat",
    "caviar": "caviar",
    "ivory carvings": "ivory", "ivory pieces": "ivory", "ivory scraps": "ivory",
    "jewellery - ivory ": "ivory", "jewellery - ivory": "ivory", "tusks": "ivory",
    "piano keys": "ivory", "sets of piano keys": "ivory",
    "horns": "horn", "horn carvings": "horn", "horn pieces": "horn", "horn products": "horn",
    "horn scraps": "horn",
    "bones": "bone", "bone pieces": "bone", "bone carvings": "bone", "bone products": "bone",
    "skeletons": "bone", "skulls": "skulls",
    "teeth": "teeth_claws", "claws": "teeth_claws",
    "raw corals": "coral_raw", "coral sand": "coral_raw",
    "shells": "shells", "carapaces": "shells", "scales": "scales",
    "feathers": "feathers", "quills": "feathers",
    "eggs": "eggs", "eggshell": "eggs",
    "hair": "hair", "hair products": "hair", "fibres": "hair",
    "fins": "fins", "fin (dried)": "fins", "fin (wet)": "fins", "swim bladders": "fins",
    "gill plates": "fins",
    "extract": "extracts_medicine", "medicine": "extracts_medicine",
    "derivatives": "extracts_medicine", "oil": "extracts_medicine",
    "powder": "extracts_medicine", "gall bladders": "extracts_medicine",
    "gall": "extracts_medicine", "musk": "extracts_medicine", "wax": "extracts_medicine",
    "genitalia": "extracts_medicine", "cosmetics": "extracts_medicine",
    "venom": "extracts_medicine",
    "carvings": "carvings_jewellery", "jewellery": "carvings_jewellery",
    "pearl": "carvings_jewellery", "pearls": "carvings_jewellery",
    "tails": "body_parts", "feet": "body_parts", "ears": "body_parts", "heads": "body_parts",
    "trunk": "body_parts", "baleen": "body_parts", "sawfish rostrum": "body_parts",
    "unspecified": "other",
    # plant terms found on a few animal records
    "flowers": "other", "wood product": "other", "thread": "other", "cultures": "other",
    "leaves": "other",
}

# --- Units --------------------------------------------------------------------------------
# (unit_group, multiplier to canonical unit). Canonical: count, kg, litre, m2, m3, m.
LEMIS_UNIT = {
    "NO": ("count", 1.0), "KG": ("kg", 1.0), "GM": ("kg", 1e-3), "MG": ("kg", 1e-6),
    "LT": ("litre", 1.0), "ML": ("litre", 1e-3), "C3": ("litre", 1e-3),
    "M2": ("m2", 1.0), "C2": ("m2", 1e-4), "M3": ("m3", 1.0),
    "MT": ("m", 1.0), "CM": ("m", 1e-2),
}
CITES_UNIT = {
    None: ("count", 1.0), "": ("count", 1.0), "Number of specimens": ("count", 1.0),
    "kg": ("kg", 1.0), "g": ("kg", 1e-3), "mg": ("kg", 1e-6),
    "l": ("litre", 1.0), "ml": ("litre", 1e-3), "cm3": ("litre", 1e-3),
    "m2": ("m2", 1.0), "cm2": ("m2", 1e-4), "m3": ("m3", 1.0),
    "m": ("m", 1.0), "cm": ("m", 1e-2),
    "pairs": ("count", 2.0), "sets": ("count", 1.0), "sides": ("count", 1.0),
    "microgrammes": ("kg", 1e-9),
    # before 2015 only (none in the scoring window)
    "lbs": ("kg", 0.45359237), "ft2": ("m2", 0.09290304),
    "bellyskins": ("count", 1.0), "backskins": ("count", 1.0), "hornback skins": ("count", 1.0),
    "(skins)": ("count", 1.0), "pieces": ("count", 1.0), "items": ("count", 1.0),
}

# --- Source / purpose ---------------------------------------------------------------------
# The shared form uses the CITES codes. LEMIS writes two of them differently; anything not in
# the shared set (including CITES X, which LEMIS has no equivalent for) becomes "unknown".
LEMIS_SOURCE_TO_CITES = {"P": "O"}          # pre-Convention
LEMIS_PURPOSE_TO_CITES = {"Y": "N"}         # reintroduction into the wild
SOURCE_CODES = {"W", "C", "F", "R", "D", "A", "I", "U", "O"}
PURPOSE_CODES = {"T", "P", "S", "H", "Z", "B", "E", "M", "Q", "N", "L", "G"}

# --- Countries ----------------------------------------------------------------------------
# LEMIS codes that never occur in CITES, mapped to CITES's own code or the ISO successor.
LEMIS_COUNTRY_TO_CITES = {"VS": "XV", "ZZ": "HS", "ZR": "CD", "TP": "TL"}

# --- Taxonomy -----------------------------------------------------------------------------
# LEMIS follows Catalogue of Life class names, CITES its own nomenclature.
LEMIS_CLASS_TO_CITES = {"Actinopterygii": "Actinopteri"}
# Where one LEMIS class covers a CITES class with another name, the genus decides:
# LEMIS "Sarcopterygii" covers CITES Coelacanthi and Dipneusti; LEMIS "Clitellata" holds both
# earthworms and leeches, and CITES files leeches as Hirudinoidea.
LEMIS_CLASS_BY_GENUS = {
    ("Sarcopterygii", "latimeria"): "Coelacanthi",
    ("Sarcopterygii", "neoceratodus"): "Dipneusti",
    ("Sarcopterygii", "protopterus"): "Dipneusti",
    ("Sarcopterygii", "lepidosiren"): "Dipneusti",
    ("Clitellata", "hirudo"): "Hirudinoidea",
    ("Clitellata", "hirudinaria"): "Hirudinoidea",
    ("Clitellata", "placobdella"): "Hirudinoidea",
}
# Placeholder "genera" in LEMIS that name no taxon; treated as unknown.
LEMIS_PLACEHOLDER_GENERA = {"Tropical fish", "All live tropical fish", "Non-CITES entry",
                            "Other live inverts", "Various fish", "Unknown", "All plants",
                            "Migratory bird", "Freshwater fish", "Fish ectoparasites"}
# Classes in LEMIS that are not animals; the model learns from animals with a class.
NON_ANIMAL_CLASSES = {"Magnoliopsida", "Liliopsida", "Polypodiopsida", "Cycadopsida",
                      "Pinopsida", "Ulvophyceae", "Phaeophyceae", "Gammaproteobacteria",
                      "Chytridiomycetes", "Agaricomycetes"}

# --- Countries -> world region (used for the simulated domain-shift experiment) -----------
REGION_OF = {}
_REGIONS = {
    "Africa": "DZ AO BJ BW BF BI CM CV CF TD KM CG CD CI DJ EG GQ ER ET GA GM GH GN GW KE LS LR "
              "LY MG MW ML MR MU MA MZ NA NE NG RW ST SN SC SL SO ZA SS SD SZ TZ TG TN UG EH ZM ZW YT RE",
    "Asia": "AF AM AZ BH BD BT BN KH CN GE HK IN ID IR IQ IL JP JO KZ KW KG LA LB MO MY MV MN MM "
            "NP KP OM PK PS PH QA SA SG KR LK SY TW TJ TH TL TR TM AE UZ VN YE",
    "Europe": "AL AD AT BY BE BA BG HR CY CZ DK EE FO FI FR DE GI GR HU IS IE IT LV LI LT LU MT MD "
              "MC ME NL MK NO PL PT RO RU SM RS SK SI ES SE CH UA GB VA XK GG JE IM",
    "Americas": "AG AR BS BB BZ BO BR CA CL CO CR CU DM DO EC SV GD GT GY HT HN JM MX NI PA PY PE "
                "KN LC VC SR TT US UY VE AI AW BM VG KY CW GF GP MQ MS PR SX TC VI BQ PM GL FK AN",
    "Oceania": "AU FJ KI MH FM NR NZ PW PG WS SB TO TV VU NC PF CK NU AS GU MP WF TK NF PN",
}
for _region, _codes in _REGIONS.items():
    for _c in _codes.split():
        REGION_OF[_c] = _region


def region(code):
    return REGION_OF.get(code, "Unknown")
