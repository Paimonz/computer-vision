{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1FZrhv2D7N22W-IdQVGReP303Fp2OQZmo",
      "authorship_tag": "ABX9TyNOerETEWxaFsAJC9Dbf/+g",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Paimonz/computer-vision/blob/main/imageclassification.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 71,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oacAu5x0TNXX",
        "outputId": "5bc4a901-5e64-4ac4-c3b4-7f7dfc2fc8db"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "TensorFlow version: 2.12.0\n",
            "TensorFlow Hub version: 0.14.0\n"
          ]
        }
      ],
      "source": [
        "import pathlib\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "import tensorflow_hub as hub\n",
        "print('TensorFlow version:', tf.__version__)\n",
        "print('TensorFlow Hub version:', hub.__version__)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model_name = 'efficientnetv2-b2-21k'"
      ],
      "metadata": {
        "id": "b9Wyt3T9WTL-"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "models_path = {\n",
        "    \"efficientnetv2-b2-21k\": \"https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/feature_vector/2\",\n",
        "}"
      ],
      "metadata": {
        "id": "rrx5IQ5tWicE"
      },
      "execution_count": 37,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "models_pixels = {\n",
        "    \"efficientnetv2-b2-21k\": 260\n",
        "}"
      ],
      "metadata": {
        "id": "GS2YK1ZBYZUW"
      },
      "execution_count": 38,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model_path = models_path.get(model_name)\n",
        "model_path\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 36
        },
        "id": "JkLdZBgWZG_d",
        "outputId": "9eb242b3-64ff-434f-b20e-7d7dfa3f5daf"
      },
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/feature_vector/2'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 39
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pixels = models_pixels.get(model_name)\n",
        "pixels"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tn_U-L3-aFqc",
        "outputId": "be933392-226e-4a25-81ec-0cca9c3f4e6a"
      },
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "260"
            ]
          },
          "metadata": {},
          "execution_count": 40
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "image_size = (pixels, pixels)\n",
        "image_size"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CfLw9ro5aAOd",
        "outputId": "4cf4cde1-161d-4038-c6d1-f2665c8583ed"
      },
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(260, 260)"
            ]
          },
          "metadata": {},
          "execution_count": 41
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print('Model: ', model_name)\n",
        "print('Path: ', model_path)\n",
        "print('Imagem size: ', image_size)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9H5a9r2nbwTE",
        "outputId": "4afd0ab1-b91f-41bc-f258-626e2df83e13"
      },
      "execution_count": 42,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model:  efficientnetv2-b2-21k\n",
            "Path:  https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/feature_vector/2\n",
            "Imagem size:  (260, 260)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "'''data_directory = tf.keras.utils.get_file('dataset','https://github.com/Paimonz/computer-vision/tree/main/dataset')\n",
        "print(data_directory)'''\n",
        "\n"
      ],
      "metadata": {
        "id": "w41GpqUvaP6k",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "66830f5d-f117-409c-d1f8-977444947cee"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading data from https://github.com/Paimonz/computer-vision/tree/main/dataset\n",
            "   8192/Unknown - 0s 0us/step/root/.keras/datasets/dataset\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "'''train_dataset = tf.keras.preprocessing.image_dataset_from_directory(data_directory,\n",
        "                                                                    validation_split= .20,\n",
        "                                                                    subset = 'training',\n",
        "                                                                    label_mode='categorical',\n",
        "                                                                    seed = 123,\n",
        "                                                                    image_size=image_size,\n",
        "                                                                    batch_size=1)'''"
      ],
      "metadata": {
        "id": "H5oRGcMFokwy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from keras.preprocessing.image import ImageDataGenerator"
      ],
      "metadata": {
        "id": "xpGy9_6RCB_D"
      },
      "execution_count": 43,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "flow_from_directory = \"/content/drive/MyDrive/Colab Notebooks/dataset\"\n",
        "data_directory = flow_from_directory\n",
        "print(data_directory)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CjJbAW3o_lFT",
        "outputId": "f22c3491-2561-4c85-b9c2-8bd3af373dce"
      },
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/drive/MyDrive/Colab Notebooks/dataset\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_dataset = tf.keras.preprocessing.image_dataset_from_directory(data_directory,\n",
        "                                                                    validation_split= .20,\n",
        "                                                                    subset = 'training',\n",
        "                                                                    label_mode='categorical',\n",
        "                                                                    seed = 123,\n",
        "                                                                    image_size=image_size,\n",
        "                                                                    batch_size=1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BHIDozoxCqki",
        "outputId": "10a358dd-1914-4f61-8ef3-43921a1ee278"
      },
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 1050 files belonging to 2 classes.\n",
            "Using 840 files for training.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_dataset.class_names"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XmmCwjL1FWm1",
        "outputId": "b1474724-7035-4285-cf80-8d44f75f23b6"
      },
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['cataract', 'normal']"
            ]
          },
          "metadata": {},
          "execution_count": 46
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "classes = train_dataset.class_names\n",
        "classes"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hBSJ21aSFr7K",
        "outputId": "3d1ae3e0-c809-4a96-d478-82bd81e93027"
      },
      "execution_count": 47,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['cataract', 'normal']"
            ]
          },
          "metadata": {},
          "execution_count": 47
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "training_size = train_dataset.cardinality().numpy()\n",
        "training_size"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ftzcCd8FGAZc",
        "outputId": "f8495af2-b02a-4ce8-8516-6a7cfa9f79d7"
      },
      "execution_count": 48,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "840"
            ]
          },
          "metadata": {},
          "execution_count": 48
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "BATCH_SIZE = 16\n"
      ],
      "metadata": {
        "id": "h_gLIlALGbNL"
      },
      "execution_count": 49,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_dataset = train_dataset.unbatch().batch(BATCH_SIZE)\n",
        "train_dataset"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ki3giSx0GlYz",
        "outputId": "35c6b5c7-c2e6-42f4-abdc-854cec15f24d"
      },
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<_BatchDataset element_spec=(TensorSpec(shape=(None, 260, 260, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 2), dtype=tf.float32, name=None))>"
            ]
          },
          "metadata": {},
          "execution_count": 50
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "840/16\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uUzbRQoCHjD7",
        "outputId": "172c9a14-ab44-4893-9a79-ed087f2b4394"
      },
      "execution_count": 51,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "52.5"
            ]
          },
          "metadata": {},
          "execution_count": 51
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_dataset = train_dataset.repeat()"
      ],
      "metadata": {
        "id": "SNIDsQJlH5yy"
      },
      "execution_count": 52,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from multiprocessing.process import parent_process\n",
        "normalization_layer = tf.keras.layers.Rescaling(1. /255)\n",
        "pre_processing = tf.keras.Sequential([normalization_layer])\n",
        "pre_processing.add(tf.keras.layers.RandomRotation(40))\n",
        "pre_processing.add(tf.keras.layers.RandomTranslation(0, 0.2))\n",
        "pre_processing.add(tf.keras.layers.RandomTranslation(0.2, 0))\n",
        "pre_processing.add(tf.keras.layers.RandomZoom(0.2, 0.2))\n",
        "pre_processing.add(tf.keras.layers.RandomFlip(mode = 'horizontal'))"
      ],
      "metadata": {
        "id": "RNOszHEMITTK"
      },
      "execution_count": 53,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_dataset = train_dataset.map(lambda images, labels: (pre_processing(images), labels))\n",
        "train_dataset"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "b08JoMZwKQ-4",
        "outputId": "a265eee5-efbd-4858-9cef-5aa35c0ea6ef"
      },
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<_MapDataset element_spec=(TensorSpec(shape=(None, 260, 260, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 2), dtype=tf.float32, name=None))>"
            ]
          },
          "metadata": {},
          "execution_count": 54
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_dataset = tf.keras.preprocessing.image_dataset_from_directory(data_directory,\n",
        "                                                                   validation_split= .20,\n",
        "                                                                   subset ='validation',\n",
        "                                                                   label_mode ='categorical',\n",
        "                                                                   seed = 123,\n",
        "                                                                   image_size = image_size,\n",
        "                                                                   batch_size = 1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KzoWG6q2LnLh",
        "outputId": "4439fae1-547c-4343-c8ef-f2bacef5cdfc"
      },
      "execution_count": 55,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 1050 files belonging to 2 classes.\n",
            "Using 210 files for validation.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_size = test_dataset.cardinality().numpy()\n",
        "test_size"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "X0lPQpzVMxMx",
        "outputId": "272c2d9f-aa7f-499c-a32e-e2ee25b38d8e"
      },
      "execution_count": 56,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "210"
            ]
          },
          "metadata": {},
          "execution_count": 56
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_dataset = test_dataset.unbatch().batch(BATCH_SIZE)\n",
        "test_dataset = test_dataset.map(lambda images, labels: (pre_processing(images), labels))"
      ],
      "metadata": {
        "id": "V-advWyENUxD"
      },
      "execution_count": 57,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = tf.keras.Sequential([\n",
        "                              tf.keras.layers.InputLayer(input_shape= image_size + (3,)),\n",
        "                              hub.KerasLayer(model_path, trainable = False),\n",
        "                              tf.keras.layers.Dropout(rate = 0.2),\n",
        "                              tf.keras.layers.Dense(len(classes))\n",
        "])"
      ],
      "metadata": {
        "id": "UwnRGQnHaE1W"
      },
      "execution_count": 58,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "(None, ) + image_size + (3, )"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NH7WmQ-Hc8Yb",
        "outputId": "481273f1-4047-48cf-f85d-0c58f6b1d25f"
      },
      "execution_count": 62,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(None, 260, 260, 3)"
            ]
          },
          "metadata": {},
          "execution_count": 62
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.build((None, )+ image_size + (3,))\n",
        "model.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jHclCDWIbQpj",
        "outputId": "082d671d-ff00-4342-c568-6161a0c77206"
      },
      "execution_count": 63,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential_4\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " keras_layer_1 (KerasLayer)  (None, 1408)              8769374   \n",
            "                                                                 \n",
            " dropout_1 (Dropout)         (None, 1408)              0         \n",
            "                                                                 \n",
            " dense_1 (Dense)             (None, 2)                 2818      \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 8,772,192\n",
            "Trainable params: 2,818\n",
            "Non-trainable params: 8,769,374\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics = 'accuracy')"
      ],
      "metadata": {
        "id": "UslZiLBhcQJz"
      },
      "execution_count": 67,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "steps_per_epoch = training_size // BATCH_SIZE\n",
        "validation_steps = test_size // BATCH_SIZE\n",
        "print(steps_per_epoch, validation_steps)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bOaUfPmMfszz",
        "outputId": "9d27c683-3818-4006-c5eb-15ab39783ce3"
      },
      "execution_count": 64,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "52 13\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "hist = model.fit(train_dataset, epochs= 4, steps_per_epoch = steps_per_epoch,\n",
        "                 validation_data = test_dataset, validation_steps = validation_steps).history"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Dz19yiGQgcLF",
        "outputId": "6b0f2321-ab0d-4cfe-b8ef-58571e1c5449"
      },
      "execution_count": 81,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/4\n",
            "52/52 [==============================] - 168s 3s/step - loss: 0.0715 - accuracy: 0.9724 - val_loss: 0.0699 - val_accuracy: 0.9712\n",
            "Epoch 2/4\n",
            "52/52 [==============================] - 168s 3s/step - loss: 0.0772 - accuracy: 0.9757 - val_loss: 0.0768 - val_accuracy: 0.9808\n",
            "Epoch 3/4\n",
            "52/52 [==============================] - 171s 3s/step - loss: 0.0775 - accuracy: 0.9745 - val_loss: 0.0976 - val_accuracy: 0.9567\n",
            "Epoch 4/4\n",
            "52/52 [==============================] - 159s 3s/step - loss: 0.0707 - accuracy: 0.9757 - val_loss: 0.0664 - val_accuracy: 0.9712\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure()\n",
        "plt.ylabel('Loss (training and validation)')\n",
        "plt.xlabel('Steps')\n",
        "plt.plot(hist['loss'], label = 'training')\n",
        "plt.plot(hist['val_loss'], label = 'testing')\n",
        "plt.legend();"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 449
        },
        "id": "RHyK4C7vllt-",
        "outputId": "07d18afc-b3ca-4a0a-cb23-126761c6d583"
      },
      "execution_count": 79,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure()\n",
        "plt.ylabel('Accuracy (training and validation)')\n",
        "plt.xlabel('Steps')\n",
        "plt.plot(hist['accuracy'], label = 'training')\n",
        "plt.plot(hist['val_accuracy'], label = 'testing')\n",
        "plt.legend();"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 449
        },
        "id": "I3I4V3ipnj5N",
        "outputId": "a2f422db-7b1b-4f3e-f4a5-37e319291b42"
      },
      "execution_count": 80,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    }
  ]
}