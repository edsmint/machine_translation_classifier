local token_embedding_dim = 100;

{
  "train_data_path": "a5-data/de_en.train",
  "validation_data_path": "a5-data/de_en.dev",
  "dataset_reader": {
    "type": "mt_classifier_reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 64, // Hyperparameter
  },
  "trainer": {
    "num_epochs": 8, // Hyperparameter
    "patience": 15,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam" // Hyperparameter
    }
  },
  "model": {
    "type": "mt_classifier",
    "en_text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": token_embedding_dim,
          //"pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/word2vec/GoogleNews-vectors-negative300.txt.gz",
          //"trainable": true,
        },
      }
    },
    "de_text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": token_embedding_dim,
          // "pretrained_file": "/cse/web/courses/cse447/19wi/assignments/resources/fasttext_german/cc.de.300.vec.gz",
          // "trainable": true,
        },
      }
    },
    "en_pos_text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 5,
        },
      }
    },
    "en_encoder": {
      "type": "lstm",
      "input_size": token_embedding_dim,
      "hidden_size": 1,
      "num_layers": 3,
      "dropout": 0.8,
      "bidirectional": true
    },
    "de_encoder": {
      "type": "lstm",
      "input_size": token_embedding_dim,
      "hidden_size": 50,
      "num_layers": 2,
      "dropout": 0.5,
      "bidirectional": true
    },
    "pos_encoder": {
      "type": "lstm",
      "input_size": 5,
      "hidden_size": 5,
      "num_layers": 4,
      "dropout": 0.5,
      "bidirectional": true
    }
  },
}
