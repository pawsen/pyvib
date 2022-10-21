{
  description = "Nix flake for developing pyvib";

  # To update all inputs:
  # $ nix flake update --recreate-lock-file

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
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
          pip
          black     # format
          debugpy   # for dap/remote debugging
          ipython
          ipdb      # ipython debugger
          # it seems pdbpp is not packaged for nixos yet
          # pdbpp     # replacement for pdb. Try to write `sticky`
          isort     # sort imports
          pyflakes  # linter
          # pytest
        ]);
        buildInputs = with pkgs; [
          pythonEnv
          # LSP
          nodePackages.pyright
          # only for PyDSTool (which is only needed to run the 2dof example)
          swig
          gfortran
        ];
        defaultPackage = pythonPackages.buildPythonPackage {
          name = "pyvib";
          format = "setuptools";
          src = ./.;
          inherit buildInputs;
        };
      in rec {
        inherit defaultPackage;
        devShell = pkgs.mkShell {
          nativeBuildInputs = [
            defaultPackage
          ] ++ buildInputs;
          # buildInputs = [
          #   pkgs.nodePackages.pyright
          # ];

          # allow pip to install into a temp directory
          # ${pwd} is the folder where `nix develop` is run from. Using git will always return the base/root dir
          # XXX: remember to change python version below
          # XXX: pwd=$() is bash syntax. Thus the pip alias does not work for fish shells
          # `nix print-dev-env` to see the dev env
          # run the shellHook manually with
          # eval $shellHook
          shellHook = ''
            pwd=$(git rev-parse --show-toplevel)
            alias pip="PIP_PREFIX='$pwd/_build/pip_packages' TMPDIR='$HOME' command pip"
            export PYTHONBREAKPOINT="ipdb.set_trace"
            export PYTHONPATH="$pwd:$pwd/_build/pip_packages/lib/python3.9/site-packages:$PYTHONPATH"
            export PATH="$pwd/_build/pip_packages/bin:$PATH"
          '';
        };
      });
}
