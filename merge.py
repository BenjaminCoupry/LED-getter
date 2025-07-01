import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

import pipeline.common as common

import ledgetter.utils.vector_tools as vector_tools
import ledgetter.utils.loading as loading
import pipeline.outputs as outputs
import jax




def main():
    args = common.parse_merge_args()
    with jax.default_device(jax.devices(args.backend)[0]):
        values, full_mask = loading.load_chuncked_values(args.paths)
        light_dict = {'light_values': {k: v for k, v in values.items() if k not in {'validity_mask'}}}
        outputs.export_values(os.path.join(args.out_path,'values'), light_dict, full_mask, values['validity_mask'])
        

if __name__ == "__main__":
    main()