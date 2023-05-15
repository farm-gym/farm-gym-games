import gym
from gym import spaces
from gym.utils.step_api_compatibility import step_api_compatibility 
import farmgym_games.game_catalogue.farm0.farm as cb
from farmgym_games.game_builder.utils import (
    farmgymobs_to_obs,
    observation_hide_final_state_of_plants,
)
import numpy as np

import os
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

class Farm0(gym.Env):
    """
    Farm0 is a very basic 1x1 farm with only one possible plant : tomato, planted in a clay ground.
    The actions are to water the field or to harvest it.
    The advised maximum episode length is 365 (as in 365 days).

    The Farm has the weather of Montpellier in France (e.g. fairly warm weather, well suited for the culture of tomato), the initial day is 100. Initially the field is healthy and contains all the nutrient necessary to the plant.

    The reward is the number of grams of harvested tomatoes.

    The condition for end of episode (self.step returns done) is that the day is >= 365 or that the field has been harvested, or that the plant is dead.

    Parameters
    ----------
    api_compatibility: boolean, default = False
        Apply compatibility to v0.21 gym api.
    Notes
    -----
    State:
        The state consists of
        ####
        * Day (from 1 to 365)
        * mean air temperature (°C)
        * min air temperature (°C)
        * max air temperature (°C)
        * rain amount (mm)
        * sun-exposure (from 1 to 5)
        * consecutive dry day (int)
        * stage of growth of the plant (int)
        * size of the plant in cm
        * number of fruits (int)
        * weight of fruits (g).

    Actions:
        The actions are :
        ####
        * doing nothing.
        * 2 levels of watering the field (1L or 5L of water)
        * harvesting
    """
    observations_txt = [
        "Day (from 1 to 365)",
        "Mean air temperature (°C)",
        "Min air temperature (°C)",
        "Max air temperature (°C)",
        "Rain amount",
        "Sun-exposure (from 1 to 5)",
        "Consecutive dry day (int)",
        "Stage of growth of the plant",
        "Number of fruits (int)",
        "Size of the plant in cm",
        "Soil wet_surface (m2.day-1)",
        "fertilizer amount (kg)",
        "Pollinators occurrence (bin)",
        "Weeds grow (nb)",
        "Weeds flowers (nb)",
        "weight of fruits",
        "microlife health index (%)",
    ]
    name = "Farm0"
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps":4}
    def __init__(self, render_mode = "rgb_array", api_compatibility = False):
        # init base classes
        gym.Env.__init__(self)
        self.api_compatibility = api_compatibility

        self.farm = cb.env()
        self.observation_space = self.farm.observation_space
        self.action_space = self.farm.action_space
        # monitoring writer
        params = {}
        self.iteration = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # initialize
        self.state = None
        self.reset()

    def reset(self):
        #### add back hide final state of plants
        observation = self.farm.reset()[0]
        if self.api_compatibility:
            return observation
        else:
            return observation, {}

    def step(self, action):
        observation, reward, is_done, _, info = self.farm.step(action)        
        #### Add back hide final state
        ####observation, _, _, _,info = self.farm.step([])
        ####observation = observation_hide_final_state_of_plants( farmgymobs_to_obs(obs1), id_of_plants_stage=7)
        if self.api_compatibility:
            return step_api_compatibility((observation, reward, False, is_done, info), output_truncation_bool=False)
        else:
            return observation, reward, False, is_done, info

    def render(self):
        image = self._render_frame()
        if self.render_mode == "human":
            cv2.imshow("Game", image)
            cv2.waitKey(1000//self.metadata['render_fps'])
        elif self.render_mode == "rgb_array":
            return image
        else:
            raise ValueError("Unsupported rendering mode")
    
    def _render_frame(self):
        max_display_actions = self.farm.rules.actions_allowed["params"]["max_action_schedule_size"]

        from PIL import Image, ImageDraw, ImageFont
        field = self.farm.fields["Field-0"]

        im_width, im_height = 64, 64
        XX = field.X+1
        YY = (field.Y
                + (int)(
                    np.ceil(
                        len(
                            [1 for e in field.entities if field.entities[e].to_thumbnailimage() != None]
                        )
                        / field.X
                    )
                ))
        font_size = im_width * XX // 10

        offsetx = im_width // 2
        offset_header = font_size * 2
        offset_sep = font_size // 2
        offset_foot = font_size * 2

        font = ImageFont.truetype(str(CURRENT_DIR) + "/rendering/Gidole-Regular.ttf", size=font_size)
        font_action = ImageFont.truetype(
            str(CURRENT_DIR) + "/rendering/Gidole-Regular.ttf",
            size=im_width * XX // 18,
        )

        
        left, top, right, bottom = font_action.getbbox("A")
        car_height = np.abs(top - bottom) * 1.33 
        offset_actions = (int)(car_height * max_display_actions + 5 * im_height // 100)

        dashboard_picture = Image.new(
            "RGBA",
            (
                im_width * XX,
                im_height * YY + offset_header + offset_sep + offset_foot + offset_actions,
            ),
            (255, 255, 255, 255),
        )
        d = ImageDraw.Draw(dashboard_picture)

        day = (int)(field.entities["Weather-0"].variables["day#int365"].value)
        day_string = "Day {:03d}".format(day)

        d.text(
            (
                dashboard_picture.width // 2 - len(day_string) * font_size // 4,
                im_height * YY + offset_header + offset_sep + offset_foot // 4 + offset_actions,
            ),
            day_string,
            font=font,
            fill="black",
            #stroke_fill="black",
        )

        # offset_field=0
        text = "Field"  # "F-"+fi[-1:]
        left, top, right, bottom = font.getbbox(text)
        width_text = (int)(np.abs(right - left))
        d.text(
            (
                offsetx + (field.X) * im_width // 2 - width_text // 2,
                offset_header // 4,
            ),
            text,
            font=font,
            fill="black",
        )

        index = 0
        for e in field.entities:
            image = field.entities[e].to_fieldimage()
            dashboard_picture.paste(image, (offsetx, offset_header), image)

            j = index // field.X
            i = index - j * field.X
            image_t = field.entities[e].to_thumbnailimage()
            if image_t != None:
                dd = ImageDraw.Draw(image_t)
                xx = offsetx + i * im_width
                yy = offset_header + field.Y * im_height + offset_sep + j * im_height
                dashboard_picture.paste(image_t, (xx, yy), image_t)
                index += 1

        offset_field_y = (
            offset_header
            + field.Y * im_height
            + offset_sep
            + ((index - 1) // field.X + 1) * im_height
        )
        d.rectangle(
            [
                (offsetx, offset_field_y),
                (
                    offsetx + field.X * im_width,
                    offset_field_y + offset_actions + im_width // 100,
                ),
            ],
            fill=(255, 255, 255, 255),
            outline=(0, 0, 0, 255),
            width=im_width // 100,
        )

        nb_a = 0
        if self.farm.last_farmgym_action:
            for a in self.farm.last_farmgym_action:
                fa_key, fi_key, entity_key, action_name, params = a
                if a[1] == fi_key and nb_a <= max_display_actions:
                    text = action_name
                    # print("DISPLAY ACTION",action_name, params)
                    if (type(params) == dict) and ("plot" in params.keys()):
                        text += " " + str(params["plot"])
                    xx_a = offsetx + im_width // 100
                    yy_a = offset_field_y + nb_a * car_height + im_width // 100
                    d.text(
                        (xx_a, yy_a),
                        text,
                        font=font_action,
                        fill="black",
                    )
                    nb_a += 1

        offsetx += (field.X + 1) * im_width


        return np.array(dashboard_picture)
    def close(self):
        cv2.destroyAllWindows()

