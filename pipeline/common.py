
import ledgetter.utils.files as files

def update_light_dict(light_dict, name=None, losses_values=None, model=None, values=None, light=None):
    new_losses = light_dict['losses'] if (losses_values is None or name is None) else light_dict['losses'] + [(files.get_new_unique_name(list(map(lambda u : u[0], light_dict['losses'])), name), losses_values)]
    light_dict = {'model': light_dict['model'] if model is None else model,
                   'light_values': light_dict['light_values'] if values is None else values,
                     'light': light_dict['light'] if light is None else light,
                       'losses' : new_losses}
    return light_dict