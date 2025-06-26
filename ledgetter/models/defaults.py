import ledgetter.models.models as models

def get_default(name, raycaster, scale):
    valid_options={'local_threshold':0.5, 'global_threshold':0.1, 'dilation':2, 'erosion':9, 'raycaster' : raycaster, 'radius' : 0.0005*scale}
    match name:
        case 'directional':
            data = {'normals', 'points', 'pixels'}
            parameters = {'rho', 'light_directions', 'dir_light_power'}
            model = {'light': 'directional', 'renderers': ['lambertian'], 'parameters': parameters, 'data':  data}
            validity_masker = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options})
            return model, validity_masker
        case 'rail':
            data = {'normals', 'points', 'center', 'light_directions', 'dir_light_power', 'pixels'}
            parameters = {'rho', 'light_distance'}
            model = {'light': 'rail', 'renderers': ['lambertian'], 'parameters': parameters, 'data': data}
            validity_masker = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options})
            return model, validity_masker
        case 'punctual':
            data = {'normals', 'points', 'pixels'}
            parameters = {'rho', 'light_locations', 'light_power'}
            model = {'light': 'punctual', 'renderers':['lambertian'], 'parameters':parameters, 'data': data}
            validity_masker = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options})
            return model, validity_masker
        case 'LED':
            data = {'normals', 'points', 'pixels'}
            parameters = {'rho', 'light_locations', 'light_power', 'light_principal_direction', 'mu'}
            model = {'light': 'LED', 'renderers':['lambertian'], 'parameters':parameters, 'data': data}
            validity_masker = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options})
            return model, validity_masker
        case 'specular':
            data = {'normals', 'points', 'pixels'}
            parameters = {'rho', 'light_locations', 'light_power', 'light_principal_direction', 'mu', 'rho_spec', 'tau_spec'}
            model ={'light':'LED', 'renderers':['lambertian', 'phong'], 'parameters':parameters, 'data':data}
            validity_masker = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options})
            return model, validity_masker
        case 'harmonic':
            data = {'normals', 'points', 'pixels', 'indices', 'l_max'}
            parameters = {'rho', 'light_locations', 'light_power', 'light_principal_direction', 'free_rotation', 'coefficients', 'rho_spec', 'tau_spec'}
            model = {'light':'harmonic', 'renderers':['lambertian', 'phong'], 'parameters':parameters, 'data':data}
            validity_masker = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options})
            return model, validity_masker
        case 'grid':
            data = {'normals', 'points', 'pixels', 'min_range', 'max_range', 'pixel_step'}
            parameters = {'rho', 'direction_grid', 'intensity_grid'}
            model = {'light':'grid', 'renderers':['lambertian'], 'parameters':parameters, 'data':data}
            validity_masker = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options})
            return model, validity_masker
        case 'PS':
            data = { 'points', 'pixels', 'light_local_direction', 'light_local_intensity'}
            parameters = {'normals', 'rho'}
            model = {'light':'constant', 'renderers':['lambertian'], 'parameters':parameters, 'data':data}
            validity_masker = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options})
            return model, validity_masker
        case _:
            raise ValueError(f"Unknown default name: {name}")