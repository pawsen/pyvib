{
  description = "Nix flake for developing pyvib";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    let out = system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          # Defines dependencies missing from nixpkgs here
          # overlays = [
          #   (import ./python-overlay.nix)
          # ];
        };
        python = pkgs.python39;
        pythonPackages = python.pkgs;

        pythonEnv = python.withPackages (ps: with ps; [
          numpy
          scipy
          matplotlib
          # Dev dependencies
          black
          debugpy
          ipython
          isort
          # pytest
        ]);
        buildInputs = with pkgs; [
          pythonEnv
        ];
        defaultPackage = pythonPackages.buildPythonPackage {
          name = "pyvib";
          format = "setuptools";
          src = ./.;
          inherit buildInputs;
        };
      in
      {
        inherit defaultPackage;
        devShell = pkgs.mkShell {
          nativeBuildInputs = [
            defaultPackage
          ] ++ buildInputs;
          buildInputs = [
            pkgs.nodePackages.pyright
          ];
          shellHook = ''
          '';
        };
      }; in with utils.lib; eachSystem defaultSystems out;
}
