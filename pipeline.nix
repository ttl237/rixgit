let
  default = import ./default.nix;
  defaultPkgs = default.pkgs;
  defaultShell = default.shell;
  defaultBuildInputs = defaultShell.buildInputs;
  defaultConfigurePhase = ''
    cp ${./_rixpress/default_libraries.py} libraries.py
    cp ${./_rixpress/default_libraries.R} libraries.R
    mkdir -p $out  
    mkdir -p .julia_depot  
    export JULIA_DEPOT_PATH=$PWD/.julia_depot  
    export HOME_PATH=$PWD
  '';
  
  # Function to create R derivations
  makeRDerivation = { name, buildInputs, configurePhase, buildPhase, src ? null }:
    defaultPkgs.stdenv.mkDerivation {
      inherit name src;
      dontUnpack = true;
      inherit buildInputs configurePhase buildPhase;
      installPhase = ''
        cp ${name} $out/
      '';
    };
  # Function to create Python derivations
  makePyDerivation = { name, buildInputs, configurePhase, buildPhase, src ? null }:
    let
      pickleFile = "${name}";
    in
      defaultPkgs.stdenv.mkDerivation {
        inherit name src;
        dontUnpack = true;
        buildInputs = buildInputs;
        inherit configurePhase buildPhase;
        installPhase = ''
          cp ${pickleFile} $out
        '';
      };

  # Define all derivations
    raw_df = makePyDerivation {
    name = "raw_df";
    src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./data/HeartDiseaseTrain-Test.csv ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src input_folder
python -c "
exec(open('libraries.py').read())
file_path = 'input_folder/data/HeartDiseaseTrain-Test.csv'
data = eval('pd.read_csv')(file_path)
with open('raw_df', 'wb') as f:
    pickle.dump(data, f)
"
    '';
  };

  encoded_df = makePyDerivation {
    name = "encoded_df";
     src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./functions.py ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src/* .
      python -c "
exec(open('libraries.py').read())
with open('${raw_df}/raw_df', 'rb') as f: raw_df = pickle.load(f)
exec(open('functions.py').read())
exec('encoded_df = encode_categoricals(raw_df)')
with open('encoded_df', 'wb') as f: pickle.dump(globals()['encoded_df'], f)
"
    '';
  };

  target_dist_plot_png = makePyDerivation {
    name = "target_dist_plot_png";
     src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./functions.py ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src/* .
      python -c "
exec(open('libraries.py').read())
with open('${encoded_df}/encoded_df', 'rb') as f: encoded_df = pickle.load(f)
exec(open('functions.py').read())
exec('target_dist_plot_png = make_target_dist_png(encoded_df)')
copy_file(globals()['target_dist_plot_png'], 'target_dist_plot_png')
"
    '';
  };

  correlation_heatmap_png = makePyDerivation {
    name = "correlation_heatmap_png";
     src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./functions.py ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src/* .
      python -c "
exec(open('libraries.py').read())
with open('${encoded_df}/encoded_df', 'rb') as f: encoded_df = pickle.load(f)
exec(open('functions.py').read())
exec('correlation_heatmap_png = make_correlation_heatmap_png(encoded_df)')
copy_file(globals()['correlation_heatmap_png'], 'correlation_heatmap_png')
"
    '';
  };

  processed_data = makePyDerivation {
    name = "processed_data";
     src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./functions.py ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src/* .
      python -c "
exec(open('libraries.py').read())
with open('${encoded_df}/encoded_df', 'rb') as f: encoded_df = pickle.load(f)
exec(open('functions.py').read())
exec('processed_data = make_processed_data(encoded_df)')
with open('processed_data', 'wb') as f: pickle.dump(globals()['processed_data'], f)
"
    '';
  };

  svm_rbf_model = makePyDerivation {
    name = "svm_rbf_model";
     src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./functions.py ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src/* .
      python -c "
exec(open('libraries.py').read())
with open('${processed_data}/processed_data', 'rb') as f: processed_data = pickle.load(f)
exec(open('functions.py').read())
exec('svm_rbf_model = train_svm_rbf(processed_data)')
with open('svm_rbf_model', 'wb') as f: pickle.dump(globals()['svm_rbf_model'], f)
"
    '';
  };

  y_pred = makePyDerivation {
    name = "y_pred";
     src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./functions.py ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src/* .
      python -c "
exec(open('libraries.py').read())
with open('${processed_data}/processed_data', 'rb') as f: processed_data = pickle.load(f)
with open('${svm_rbf_model}/svm_rbf_model', 'rb') as f: svm_rbf_model = pickle.load(f)
exec(open('functions.py').read())
exec('y_pred = predict_labels(svm_rbf_model, processed_data)')
with open('y_pred', 'wb') as f: pickle.dump(globals()['y_pred'], f)
"
    '';
  };

  accuracy = makePyDerivation {
    name = "accuracy";
     src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./functions.py ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src/* .
      python -c "
exec(open('libraries.py').read())
with open('${processed_data}/processed_data', 'rb') as f: processed_data = pickle.load(f)
with open('${y_pred}/y_pred', 'rb') as f: y_pred = pickle.load(f)
exec(open('functions.py').read())
exec('accuracy = compute_accuracy(processed_data, y_pred)')
with open('accuracy', 'wb') as f: pickle.dump(globals()['accuracy'], f)
"
    '';
  };

  evaluation_df = makePyDerivation {
    name = "evaluation_df";
     src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./functions.py ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src/* .
      python -c "
exec(open('libraries.py').read())
with open('${processed_data}/processed_data', 'rb') as f: processed_data = pickle.load(f)
with open('${y_pred}/y_pred', 'rb') as f: y_pred = pickle.load(f)
exec(open('functions.py').read())
exec('evaluation_df = make_evaluation_df(processed_data, y_pred)')
with open('evaluation_df', 'wb') as f: pickle.dump(globals()['evaluation_df'], f)
"
    '';
  };

  evaluation_csv = makePyDerivation {
    name = "evaluation_csv";
     src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./functions.py ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src/* .
      python -c "
exec(open('libraries.py').read())
with open('${evaluation_df}/evaluation_df', 'rb') as f: evaluation_df = pickle.load(f)
exec(open('functions.py').read())
exec('evaluation_csv = evaluation_df')
write_to_csv(globals()['evaluation_csv'], 'evaluation_csv')
"
    '';
  };

  evaluation_df_r = makeRDerivation {
    name = "evaluation_df_r";
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      Rscript -e "
        source('libraries.R')
        evaluation_csv <- read.csv('${evaluation_csv}/evaluation_csv')
        evaluation_df_r <- evaluation_csv
        saveRDS(evaluation_df_r, 'evaluation_df_r')"
    '';
  };

  eval_factors = makeRDerivation {
    name = "eval_factors";
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      Rscript -e "
        source('libraries.R')
        evaluation_csv <- read.csv('${evaluation_csv}/evaluation_csv')
        eval_factors <- mutate(evaluation_csv, across(everything(), factor))
        saveRDS(eval_factors, 'eval_factors')"
    '';
  };

  confusion_matrix = makeRDerivation {
    name = "confusion_matrix";
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      Rscript -e "
        source('libraries.R')
        eval_factors <- readRDS('${eval_factors}/eval_factors')
        confusion_matrix <- conf_mat(eval_factors, truth, estimate)
        saveRDS(confusion_matrix, 'confusion_matrix')"
    '';
  };

  confusion_matrix_plot_png = makeRDerivation {
    name = "confusion_matrix_plot_png";
     src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./functions.R ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src/* .
      Rscript -e "
        source('libraries.R')
        confusion_matrix <- readRDS('${confusion_matrix}/confusion_matrix')
        source('functions.R')
        confusion_matrix_plot_png <- save_confusion_plot(confusion_matrix)
        copy_file_r(confusion_matrix_plot_png, 'confusion_matrix_plot_png')"
    '';
  };

  # Generic default target that builds all derivations
  allDerivations = defaultPkgs.symlinkJoin {
    name = "all-derivations";
    paths = with builtins; attrValues { inherit raw_df encoded_df target_dist_plot_png correlation_heatmap_png processed_data svm_rbf_model y_pred accuracy evaluation_df evaluation_csv evaluation_df_r eval_factors confusion_matrix confusion_matrix_plot_png; };
  };

in
{
  inherit raw_df encoded_df target_dist_plot_png correlation_heatmap_png processed_data svm_rbf_model y_pred accuracy evaluation_df evaluation_csv evaluation_df_r eval_factors confusion_matrix confusion_matrix_plot_png;
  default = allDerivations;
}
