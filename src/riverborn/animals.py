
import importlib
import json
from pathlib import Path

from pyglm import glm

from .scene import Material, Instance


ANIMAL_Y = {
    'hippo': 0.0,
    'crocodile': 1.0,
    'flamingo': 1.0,
}


class Animals:
    def __init__(self, scene):
        self.scene = scene
        self.animal_models = {}
        self.animals = {}

        default_material = Material(
            double_sided=False,
            translucent=False,
            transmissivity=0.0,
            receive_shadows=True,
            cast_shadows=True,
            alpha_test=False,
        )

        # Load animal models
        self.animal_models = {
            'hippo': self.scene.load_wavefront('hippopotamus.obj', material=default_material, capacity=3),
            'crocodile': self.scene.load_wavefront('crocodile.obj', material=default_material, capacity=8),
            'flamingo': self.scene.load_wavefront('flamingo.obj', material=default_material, capacity=15),
        }

    def load(self):
        animal_data = importlib.resources.files('riverborn') / "data/animals.json"
        try:
            with animal_data.open('r') as f:
                animals = json.load(f)
        except FileNotFoundError:
            animals = {}

        self.animals = {}

        for animal, instances in animals.items():
            self.animals[animal] = []
            for obj in instances:
                animal_model = self.scene.add(self.animal_models[animal])
                px, pz = obj['pos']
                rot = obj['rot']
                animal_model.pos = glm.vec3(px, ANIMAL_Y[animal], pz)
                animal_model.rot = glm.quat(glm.angleAxis(rot, glm.vec3(0, 1, 0)))
                self.animals[animal].append(animal_model)

    def save(self):
        animals_path = Path(__file__).parent / 'data/animals.json'
        animals_path.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        for animal, instances in self.animals.items():
            data[animal] = []
            for instance in instances:
                pos = (instance.pos.x, instance.pos.z)
                rot = glm.angle(instance.rot)
                data[animal].append({'pos': pos, 'rot': rot})

        # Save the animals' positions to a file
        with animals_path.open('w') as f:
            json.dump(data, f)

    def add(self, animal, pos, rot) -> Instance:
        inst = self.scene.add(self.animal_models[animal])
        inst.pos = glm.vec3(pos[0], ANIMAL_Y[animal], pos[2])
        inst.rot = glm.quat(glm.angleAxis(rot, glm.vec3(0, 1, 0)))
        self.animals.setdefault(animal, []).append(inst)
        return inst
