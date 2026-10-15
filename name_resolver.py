"""Name matching used by the original fixed-camera experiment."""
import re

SYN = {
    "tv": [["tv", "television"], ["screen"]],
    "television": [["tv", "television"], ["screen"]],
    "sofa": [["sofa", "couch"], ["settee"]],
    "couch": [["sofa", "couch"]],
    "chair": [["chair"], ["stool", "seat"]],
    "stool": [["stool"], ["chair", "seat"]],
    "desk chair": [["desk chair", "office chair", "chair", "seat"], ["desk"]],
    "table": [["table"], ["desk", "counter", "island"]],
    "desk": [["desk"], ["table"]],
    "bed": [["bed"], ["mattress"]],
    "stove": [["stove", "hob", "cooktop", "cooker", "oven", "burner", "range", "grate"]],
    "oven": [["oven"], ["stove"]],
    "refrigerator": [["refrigerator", "fridge", "freezer"]],
    "fridge": [["fridge", "refrigerator", "freezer"]],
    "washer": [["washer", "washing machine", "laundry"]],
    "washing machine": [["washing machine", "washer", "laundry"]],
    "dishwasher": [["dishwasher", "dish washer"], ["dish"]],
    "dryer": [["dryer", "laundry"]],
    "heater": [["heater", "radiator", "heating"]],
    "radiator": [["radiator", "heater"]],
    "fireplace": [["fireplace", "chimney", "hearth"]],
    "toilet": [["toilet"]],
    "bathtub": [["bathtub", "bath tub", "tub"], ["bath"]],
    "bathtube": [["bathtub", "bath tub", "tub"], ["bath"]],
    "sink": [["sink", "washbasin", "basin", "vanity"]],
    "shower": [["shower"]],
    "door": [["door"]],
    "doorframe": [["doorframe", "door frame"], ["door"]],
    "window": [["window"], ["glazing"]],
    "mirror": [["mirror"]],
    "ceiling light": [["ceiling light", "ceiling lamp", "ceiling fixture", "chandelier", "ceiling opal"],
                      ["ceiling"]],
    "lamp": [["lamp"], ["light fixture", "ceiling", "light"]],
    "fan": [["fan"]],
    "monitor": [["monitor"], ["screen", "display"]],
    "computer tower": [["tower", "computer case", "pc case", "desktop"]],
    "keyboard": [["keyboard"]],
    "computer mouse": [["computer mouse"], ["mouse"]],
    "mouse": [["mouse"]],
    "telephone": [["telephone", "phone"], ["handset"]],
    "phone": [["phone", "telephone"]],
    "printer": [["printer"]],
    "power strip": [["power strip", "power socket", "socket", "outlet", "power outlet"]],
    "whiteboard": [["whiteboard", "white board"], ["board"]],
    "bookshelf": [["bookshelf", "book shelf", "bookcase", "shelving unit"], ["shelf", "cabinet"]],
    "shelf": [["shelf", "shelving"]],
    "trash can": [["trash can", "trash bin", "trash", "bin", "wastebasket", "waste", "garbage", "rubbish"],
                  ["basket"]],
    "trash bin": [["trash bin", "trash can", "trash", "bin", "wastebasket", "waste", "garbage", "rubbish"],
                  ["basket"]],
    "backpack": [["backpack", "back pack", "rucksack"], ["bag"]],
    "paper bag": [["paper bag"], ["bag"]],
    "pillow": [["pillow"], ["cushion"]],
    "blanket": [["blanket", "duvet", "comforter", "quilt"]],
    "towel": [["towel"]],
    "basket": [["basket"]],
    "bowl": [["bowl"]],
    "cup": [["cup", "mug", "glass", "tumbler"]],
    "mug": [["mug", "cup"]],
    "pan": [["pan", "skillet", "frying"]],
    "pot": [["pot", "saucepan"]],
    "kettle": [["kettle", "teapot", "tea pot"], ["pot"]],
    "cutting board": [["cutting board", "chopping board", "cutting", "chopping"]],
    "microwave": [["microwave"]],
    "nightstand": [["nightstand", "night stand", "bedside table", "bedside cabinet", "bedside",
                    "night table", "end table", "side table"], ["table"]],
    "clock": [["clock"], ["watch"]],
    "plant": [["plant", "potted", "pot plant"], ["vase"]],
    "vase": [["vase"]],
    "suitcase": [["suitcase", "luggage"], ["case", "box"]],
    "crate": [["crate"], ["box", "container"]],
    "box": [["box", "container", "carton"]],
    "shoes": [["shoe"], ["sneaker", "boot"]],
    "shoe rack": [["shoe rack", "shoe shelf", "shoe"], ["rack"]],
    "coat rack": [["coat rack", "coat hook", "coat hooks", "coat"], ["rack", "hook"]],
    "candle": [["candle"]],
    "candles": [["candle"]],
    "trophy": [["trophy", "trophies", "award"]],
    "trophies": [["trophy", "trophies", "award"]],
    "recycling bin": [["recycling"], ["trash", "bin", "waste"]],
    "bucket": [["bucket", "pail"]],
    "stairs": [["stair"], ["step"]],
    "paper shredder": [["paper shredder", "shredder"], ["shredder"]],
    "shredder": [["shredder"]],
    "headphone": [["headphone", "headset", "earphone"]],
    "headphones": [["headphone", "headset", "earphone"]],
    "dish": [["dish", "plate"]],
    "dishes": [["dish", "plate"]],
    "counter": [["counter", "countertop", "worktop"], ["island"]],
    "island": [["island"]],
    "kitchen island": [["island"], ["counter", "countertop", "worktop", "peninsula"], ["kitchen"]],
    "wardrobe": [["wardrobe", "closet", "armoire"], ["closet"]],
    "closet": [["closet", "wardrobe"]],
    "column": [["column", "pillar", "post"]],
    "computer": [["computer", "desktop", "pc"]],
    "console": [["console"]],
    "speaker": [["speaker"]],
    "painting": [["painting", "picture", "frame", "art"]],
    "picture": [["picture", "frame", "art"]],
    "poster": [["poster"]],
    "rug": [["rug", "carpet", "mat"]],
    "curtain": [["curtain", "blind", "drape"]],
    "blinds": [["blind", "curtain"]],
    "bin": [["bin", "trash", "garbage", "waste"]],
    "white wall": [["wall"]],
    "green chair": [["green chair", "chair"], ["green"]],
    "desk with monitor": [["desk", "monitor"], ["desk"]],
}

FURNITURE_WORDS = {"cabinet", "cupboard", "wardrobe", "pantry", "oven", "microwave", "fridge",
                   "freezer", "shower", "washing", "dryer", "dresser", "sideboard", "hutch",
                   "nightstand", "locker", "boiler", "car", "van", "pet", "sink", "dishwasher"}

COMPOUND_BLOCK = {"door": FURNITURE_WORDS}

DROP_COLOR = {"green", "blue", "red", "black", "white", "grey", "gray", "brown", "yellow",
              "orange", "silver", "dark", "light", "pink", "beige", "wooden", "wood"}

DROP_LEAD = {"the", "a", "an"} | DROP_COLOR

CUT_WORDS = {"with", "on", "of", "closest", "closer", "opposite", "positioned", "beside",
             "next", "near", "behind", "under", "inside", "top", "at", "in", "by", "that",
             "which", "and"}

def head_term(phrase, keep_color=False):
    s = phrase.strip().strip(".,?")
    s = re.sub(r"\s+\d+$", "", s)
    drop = ({"the", "a", "an"} if keep_color else DROP_LEAD)
    words = s.split()
    while words and words[0].lower().strip("'") in drop:
        words.pop(0)
    out = []
    for w in words:
        if w.lower() in CUT_WORDS:
            break
        out.append(w)
    return " ".join(out).strip()
