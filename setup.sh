# To successfully run this code on campus clusters, please follow these steps:

# Force to buikld setuptools in conda
conda install setuptools --force-reinstall

# Avoid torchaudio version conflict
pip uninstall torchaudio