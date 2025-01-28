from models import PINN, QRes, FLS, KAN, PINNsFormer, PINNsFormer_Enc_Only, PINNMamba


def get_model(args):
    model_dict = {
        'PINN': PINN,
        'QRes': QRes,
        'FLS': FLS,
        'KAN': KAN,
        'PINNsFormer': PINNsFormer,
        'PINNMamba': PINNMamba,
        'PINNsFormer_Enc_Only': PINNsFormer_Enc_Only, # more efficient and with better performance than original PINNsFormer
    }
    return model_dict[args.model]