
from analog_gauge_reader import GaugeOption

def test_read_GaugeOption():
    json_text = """
    {
        "name": "Vorlauf",
        "rect": {
            "x": 814,
            "y": 128,
            "width": 518,
            "height": 503
        },
        "angles": {
            "min": 50,
            "max": 317
        },
        "values": {
            "min": 0,
            "max": 120
        },
        "location": "Bockholz 2",
        "unit": "Celsius",
        "rotation": "ROTATE_180"
    }"""

    opt = GaugeOption.from_json(json_text)

    assert opt is not None
    assert opt.name == 'Vorlauf'
    assert opt.angles.min == 50
    assert opt.angles.max == 317
    assert opt.values.min == 0
    assert opt.values.max == 120
    assert opt.location == 'Bockholz 2'
    assert opt.unit == 'Celsius'
    assert opt.rotation == 'ROTATE_180'


def test_read_array_of_GaugeOption():
    json_text = """
    [
        {
            "name": "Vorlauf",
            "rect": {
                "x": 814,
                "y": 128,
                "width": 518,
                "height": 503
            },
            "angles": {
                "min": 50,
                "max": 317
            },
            "values": {
                "min": 0,
                "max": 120
            },
            "location": "Bockholz 2",
            "unit": "Celsius",
            "rotation": "ROTATE_180"
        },
        {
            "name": "Ruecklauf",
            "rect": {
                "x": 368,
                "y": 363,
                "width": 505,
                "height": 495
            },
            "angles": {
                "min": 40,
                "max": 315
            },
            "values": {
                "min": 0,
                "max": 120
            },
            "location": "Bockholz 2",
            "unit": "Celsius",
            "rotation": "ROTATE_180"
        }
    ]"""

    opt: list[GaugeOption] = GaugeOption.schema().loads(json_text, many=True)

    assert opt is not None
    assert len(opt) == 2

    assert opt[0].name == 'Vorlauf'
    assert opt[0].rect.x == 814
    assert opt[0].rect.y == 128
    assert opt[0].rect.width == 518
    assert opt[0].rect.height == 503
    assert opt[0].angles.min == 50
    assert opt[0].angles.max == 317
    assert opt[0].values.min == 0
    assert opt[0].values.max == 120
    assert opt[0].location == 'Bockholz 2'
    assert opt[0].unit == 'Celsius'
    assert opt[0].rotation == 'ROTATE_180'

    assert opt[1].name == 'Ruecklauf'
    assert opt[1].rect.x == 368
    assert opt[1].rect.y == 363
    assert opt[1].rect.width == 505
    assert opt[1].rect.height == 495
    assert opt[1].angles.min == 40
    assert opt[1].angles.max == 315
    assert opt[1].values.min == 0
    assert opt[1].values.max == 120
    assert opt[1].location == 'Bockholz 2'
    assert opt[1].unit == 'Celsius'
    assert opt[1].rotation == 'ROTATE_180'
