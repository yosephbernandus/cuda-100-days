{
  description = "cuda development environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
      in with pkgs; {
        devShells.default = mkShell {
          nativeBuildInputs = [ cmake ninja gnumake pkg-config ];
          buildInputs = [ 
            cudaPackages.cudatoolkit
            # Include these if you need them
            zluda
            rocmPackages.clr
          ];
          packages = with pkgs; [ gcc12 ];
          shellHook = ''
            export CUDA_PATH=${cudaPackages.cudatoolkit}
            export LD_LIBRARY_PATH=${zluda}/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${rocmPackages.clr}/lib/:$LD_LIBRARY_PATH
          '';
        };
      });
}


