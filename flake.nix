# A Nix flake with Torch to create an impure virtual environment.

# Shell oneliner:
# nix develop --impure --expr 'let pkgs = (builtins.getFlake \
# "nixpkgs").legacyPackages.x86_64-linux; in pkgs.mkShell { name = \
# "ncaImpurePythonEnv"; venvDir = "./.venv"; buildInputs = [ pkgs.python3 \
# (pkgs.python3.withPackages (p: with p; [ click numpy torch torchvision \
# matplotlib scikit-learn pillow tqdm ])) pkgs.python3Packages.venvShellHook ]; \
# postVenvCreate = "unset SOURCE_DATE_EPOCH\npip install -r requirements.txt"; \
# postShellHook = "unset SOURCE_DATE_EPOCH"; }'
{
  inputs.nixpkgs.url = "flake:nixpkgs";

  outputs = { self, nixpkgs }: let
    pkgs = nixpkgs.legacyPackages."x86_64-linux";
    # IMPORTANT: Remember to re-create the virtual environment when switching
    # python versions or torch will complain that it imported the wrong C libs!
    #pythonPackages = pkgs.python310Packages;
    pythonPackages = pkgs.python311Packages;
    #pythonPackages = pkgs.python312Packages;
  in
  {
    # Run this with `nix develop`!
    devShell."x86_64-linux" = pkgs.mkShell {
      name = "ncaModelsImpurePythonEnv";
      venvDir = "./.venv";
      buildInputs = [
        pythonPackages.python
        pythonPackages.venvShellHook
        (pythonPackages.python.withPackages (p: with p; [
          tqdm
          click
          numpy
          pandas
          pillow
          imageio
          matplotlib
          scikit-learn
          scikit-image

          torch # or: torchWithCuda
          torchvision

          # Optional, just for experiments:
          openvino
        ]))
      ];

      # Run this command only once just after creating the virtual environment:
      postVenvCreation = ''
        unset SOURCE_DATE_EPOCH
        pip install -r requirements.txt
      '';

      # Allow pip to install wheels:
      postShellHook = ''
        unset SOURCE_DATE_EPOCH
      '';
    };
  };
}
