# store utility functions here
import gymnasium as gym
import base64
import glob
import io
from IPython import display as ipythondisplay
from IPython.display import HTML

from solarfocuser.src.controller import Controller


def show_video(prefix: str):
    """
    Display a video in a Jupyter Notebook

    Arguments
    ---------
    prefix: str

    """
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4list = [name.strip('video/') for name in mp4list]
        valid_videos = [name for name in mp4list if name.startswith(prefix)]
        if len(valid_videos) == 0:
            raise FileNotFoundError(f"Did not find a video starting with '{prefix}'. Found: {mp4list}")
        if len(valid_videos) > 1:
            raise ValueError(f"Found multiple videos starting with '{prefix}', please be more specific! Found: {valid_videos}")
        mp4 = valid_videos[0]  # we should only have one
        video = io.open('video/' + mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                    </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Did not find any files ending with .mp4 in the video folder!")


def simulate_solar_focuser(controller: Controller, video_name: str = "solar_focuser_simulation.mp4"):
    """
    Simulate the solar focuser environment and return the final state after a series of actions.
    """

    env = gym.make('SolarFocuser-v0')
    obs, info = env.reset(seed=0)
    if video_name is not None:
        env = gym.wrappers.RecordVideo(env, 'video', episode_trigger = lambda x: True,
                                       name_prefix=video_name)
    while True:
        action = controller.get_action(obs, env=env.unwrapped)  # Replace with your action selection logic
        next_obs, reward, terminated, _, info = env.step(action)
        if terminated:
            break
        obs = next_obs
    env.close()
