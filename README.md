Ce repo contient les projets implémentés pendant le cours [Intelligence Artificielle de A à Z](https://www.udemy.com/intelligence-artificielle-az/?couponCode=WEBSITE) en utilisant `PyTorch`.

1. [Installation des modules - Self Driving Car](#installation-des-modules-self-driving-car)
    1. [Installation de PyTorch](#installation-de-pytorch)
    2. [Installation de Kivy](#installation-de-kivy)
    3. [Tester votre installation](#tester-votre-installation)
2. [Installation des modules - Doom](#installation-des-modules-doom)
    1. [Installer OpenAI Gym](#installer-openai-gym)
    2. [Valider votre installation](#valider-votre-installation)
3. [Installation des modules - Breakout](#installation-des-modules-breakout)
    1. [Installer un nouvel environnement](#installer-un-nouvel-environnement)
4. [Voiture autonome - Lectures Additionnelles](#voiture-autonome-lectures-additionnelles)
5. [Doom - Lectures Additionnelles](#doom-lectures-additionnelles)
6. [Breakout - Lectures Additionnelles](#breakout-lectures-additionnelles)
7. [F.A.Q](#faq)
    1. [pip3 is not recognized as an internal or external command](#pip3-is-not-recognized-as-an-internal-or-external-command)
    2. [distributed 1.22.0 requires msgpack, which is not installed](#distributed-1220-requires-msgpack-which-is-not-installed)
    3. [tensorflow 1.9.0 has requirement setuptools=39.1.0](#tensorflow-190-has-requirement-setuptools3910)
    4. [No module named 'ai'](#no-module-named-ai)
    5. [No module named 'torch'](#no-module-named-torch)
    6. [No module named 'kivy'](#no-module-named-kivy)
    7. [No module named 'matplotlib'](#no-module-named-matplotlib)
    8. [L'installation ne fonctionne pas sous Windows](#linstallation-ne-fonctionne-pas-sous-windows)
    9. [Est-il possible de récupérer les PPT ?](#est-il-possible-de-récupérer-les-ppt-)

## Installation des modules - Self Driving Car

Après avoir installé [Anaconda](https://anaconda.org/), suivre les instructions suivantes :

### Installation de PyTorch

L'installation de PyTorch dépend elle aussi de votre système d'exploitation mais est très simple.

Rendez-vous sur https://pytorch.org/ puis choisissez :

* Votre système d'exploitation : Linux, Mac, ou Windows, selon ce que vous utilisez. 
* Votre package manager : Ici, comme nous avons téléchargé Anaconda, vous choisissez **conda**.
* Python : Nous avons installé Python **3.6**
* CUDA : Si vous avez un GPU compatible avec CUDA, vous pouvez choisir la version de CUDA associée. Si vous n'en avez pas ou si vous ne savez pas, choisissez **None**.

Vous obtiendrez alors les commandes à taper dans la fenêtre juste en dessous.

**Où taper ces commandes ?**

Si vous êtes sous Linux ou Mac, ouvrez simplement en terminal et copier/coller les commandes.

Si vous êtes sous Windows, alors :

* Dans le menu démarrer, chercher "Anaconda Prompt".
* Ouvrez l'application en faisant un clique-droit et "Lancer en tant qu'administrateur". Une console va s'ouvrir.
* Dans cette console, vous pouvez copier/coller les commandes.

**Il est recommandé de créer un environnement Anaconda au préalable.** Un environnement permet d'installer des modules spécifiquement au projet sur lequel au travail, afin d'éviter les conflits entre différentes versions. Par exemple, si vous avez aussi suivi le cours Deep Learning de A à Z, il est important de séparer les deux projets dans deux environnements différents.

Pour créer un environnement, taper les commandes suivantes dans la console **avant d'installer PyTorch** :

```
conda create --name artificialintelligenceaz python=3.6 anaconda
conda activate artificialintelligenceaz
conda install pytorch-cpu torchvision-cpu -c pytorch
spyder
```

La première ligne crée l'environnement sous Python 3.6.

La deuxième ligne active l'environnement. **Vous devrez activer l'environnement à chaque fois que vous reprenez votre travail dans ce projet.**

La troisième ligne dépend de votre OS (Mac, Linux, ou Windows), utilisez le site de PyTorch pour avoir la commande exacte.

La quatrième ligne lance Spyder. Il est important de le lancer à partir de la console pour bien rester dans l'environnement que vous venez de créer.

### Installation de Kivy

Kivy vous donne aussi les instructions séparément selon votre système d'exploitation.

Rendez-vous sur le site de Kivy, https://kivy.org/#download et cliquer sur le lien d'instructions correspondant à Windows, Mac ou Linux.

**Important : On n'utilise pas conda pour l'installation de Kivy. Elle est incompatible avec PyTorch à l'heure actuelle.**

N'oubliez pas d'activer l'environnement si vous vous êtes arrêtés entre temps :

```
conda activate artificialintelligenceaz
```

**Notes pour Windows**

Les instructions sont à l'adresse https://kivy.org/docs/installation/installation-windows.html

Elles sont résumées ici **mais pourraient être amenées à changer** dans le futur :

```
python -m pip install --upgrade pip wheel setuptools
python -m pip install docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew
python -m pip install kivy.deps.gstreamer
python -m pip install kivy.deps.angle
python -m pip install kivy
python -m pip install pygame
```

La première étape consiste à mettre à jour `pip` , `wheel` , et `setuptools`. Il est possible que mettre à jour `setuptools` casse l'installation de PyTorch. Si jamais c'est le cas, réinstallez une version précédente avec :

```
pip install setuptools==39.1.0   
```

**Notes pour Mac**

Les instructions sont à l'adresse https://kivy.org/docs/installation/installation-osx.html#using-homebrew-with-pip

Je recommande d'utiliser la section Using Homebrew with pip qui vous simplifiera grandement la vie. Si vous n'avez pas Homebrew, vous pouvez l'installer en suivant les instructions sur le site https://brew.sh/

Dans la console, tapez :

```
brew install pkg-config sdl2 sdl2_image sdl2_ttf sdl2_mixer gstreamer
pip install Cython==0.28.3
pip install kivy
pip install pygame
```

**Notes pour Linux**

Les instructions sont à l'adresse https://kivy.org/docs/installation/installation-linux.html

Dans la console, tapez :

```
sudo add-apt-repository ppa:kivy-team/kivy
sudo apt-get install python3-kivy

sudo apt-get install -y \
    python-pip \
    build-essential \
    git \
    python3 \
    python3-dev \
    ffmpeg \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    zlib1g-dev

sudo pip install --upgrade pip virtualenv setuptools
pip install Cython==0.28.2
pip install kivy
pip install pygame
```

Et voilà !

### Tester votre installation

Pour tester votre installation, taper python dans la console puis rentrez les commandes suivantes :

```
import kivy
import torch
```

Voici les sorties que vous devriez voir :

```
Python 3.6.8 |Anaconda, Inc.| (default, Dec 30 2018, 01:22:34) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import kivy
[INFO   ] [Logger      ] Record log in /home/charles/.kivy/logs/kivy_19-01-26_4.txt
[INFO   ] [Kivy        ] v1.10.1
[INFO   ] [Python      ] v3.6.8 |Anaconda, Inc.| (default, Dec 30 2018, 01:22:34) 
[GCC 7.3.0]
>>> import torch
```

## Installation des modules - Doom

### Installer OpenAI Gym

`OpenAI Gym` est à installer pour les parties 2 et 3 du cours.

Ouvrir un terminal et taper les commandes suivantes :

```
conda activate artificialintelligenceaz
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

**Installer l'environnement :**

Pour les utilisateurs de Mac :

```
brew install cmake boost boost-python sdl2 swig wget
```

Pour les utilisateurs de Ubuntu :

```
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```

**Installer Vizdoomgym :**

```
git clone https://github.com/simontudo/vizdoomgym.git
cd vizdoomgym
pip install -e .
```

### Valider votre installation

Pour être sûr que tout fonctionne, ouvrir `python` à partir de la console et tenter les commandes suivantes :

```
import gym
import vizdoomgym
env = gym.make('VizdoomBasic-v0')
```

Vous devriez ne pas avoir de message d'erreur.

Si vous en obtenez un, retentez les étapes ci-dessus et regardez la FAQ ci-dessous pour obtenir de l'aide.

## Installation des modules - Breakout

### Installer un nouvel environnement

Pour Breakout, on va devoir créer un nouvel environnement Anaconda sous Python 2.7 afin de s'assurer de la compatibilité avec les modules utilisés.

Dans la console, sous Mac ou Ubuntu :

```
conda create -n breakout python=2.7
conda activate breakout
conda install spyder
conda install -c pytorch pytorch-cpu=0.3.1
pip install gym==0.7.4
conda install -c menpo opencv=2.4.11
pip install atari-py==0.0.21
conda install -c conda-forge ffmpeg=3.2.4
spyder
```

**Note** : Assurez-vous bien de toujours activer l'environnement (`conda activate breakout`) avant de lancer `spyder`.

## Voiture autonome - Lectures additionnelles

Le premier projet va vous apprendre à créer une IA conduisant une voiture autonome simple dans un petit jeu.

**Lectures additionnelles :**

* Arthur Juliani, 2016, [Simple Reinforcement Learning with Tensorflow (Part 4)](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df)
* Tom Schaul et al., [Google DeepMind, 2016, Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
* Michel Tokic, 2010, [Adaptive ε-greedy Exploration in Reinforcement Learning Based on Value Differences](http://tokic.com/www/tokicm/publikationen/papers/AdaptiveEpsilonGreedyExploration.pdf)

## Doom - Lectures additionnelles

Dans le deuxième chapitre, nous allons créer une IA utilisant l'image afin de jouer à Doom.

**Lectures additionnelles :**

* Richard S. Sutton and Andrew G. Barto, 1998, [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning)
* Volodymyr Mnih et al., 2016, [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)

## Breakout - Lectures additionnelles

Dans ce dernier chapitre, l'IA va non seulement utiliser l'image mais aussi la composante temporelle pour jouer au casse-briques !

**Lectures additionnelles :**

* Volodymyr Mnih et al, 2016  [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
* Jaromír Janisch, 2017 [Let’s Make An A3c: Implementation](https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/)
* John Schulman et al., 2016 [High-dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf)
* Arthur Juliani, 2016 [Simple Reinforcement Learning with Tensorflow (Part 8)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)

## F.A.Q.

### pip3 is not recognized as an internal or external command

Si vous obtenez l'erreur `'pip3' is not recognized as an internal or external command`, utilisez `pip` au lieu de `pip3`.

### distributed 1.22.0 requires msgpack, which is not installed 

Si vous obtenez l'erreur `distributed 1.22.0 requires msgpack, which is not installed`, alors installez msgpack avec la ligne suivante :

`conda install -c conda-forge msgpack-python`

### tensorflow 1.9.0 has requirement setuptools<=39.1.0

Si vous obtenez l'erreur `tensorflow 1.9.0 has requirement setuptools<=39.1.0, but you'll have setuptools 39.2.0 which is incompatible`, alors il vous faut installer une version de `setuptools` inférieure.

Vous pouvez le faire avec la commande suivante :

`conda install setuptools=39.1.0`

### No module named 'ai'

Ça veut dire que vous n'êtes pas dans le bon répertoire de travail.

Dans Spyder, vous pouvez voir votre répertoire de travail dans la barre en haut à droite.

Pour changer de répertoire de travail, utilisez l'Explorateur de fichiers, et double-cliquer sur le dossier dans lequel vous voulez vous mettre. Simplement cliquer sur les petites flèches déroulantes ne suffit pas, il faut bien double-cliquer sur le dossier.

Vous pouvez aussi changer le répertoire de travail dans la console de Spyder directement en tapant :

```
import os
os.chdir("/path/to/dir")
```

### No module named 'torch'

Ça veut dire que PyTorch n'est pas installé.

En premier, vérifiez si vous êtes bien dans l'environnement que vous avez créé avec Anaconda (cf les instructions d'installation).

Si oui, alors vérifiez avec `conda list` la liste des modules installés. PyTorch ne devrait pas y être si vous avez ce message d'erreur.

Refaites alors simplement les instructions d'installation pour installer PyTorch et assurez-vous qu'il n'y a pas d'erreur dans la console. Ensuite, lancez `spyder` à partir de la console.

### No module named 'kivy'

Si l'installation de `kivy` s'est bien passée et que vous avez réussi à le lancer dans la console `python` mais que vous obtenez le message `No module named 'kivy'` dans Spyder, ça veut dire que Spyder n'est pas installé dans votre environnement.

Pour réparer ce problème :

`conda install spyder`

### No module named 'matplotlib'

Si vous obtenez ce message dans Spyder, `No module named 'matplotlib'`, alors c'est que ce module n'est pas installé.

Pour l'installer :

`conda install matplotlib`

### L'installation ne fonctionne pas sous Windows

Les modules utilisés dans les chapitres 2 et 3 ne sont pas compatibles avec Windows. Nous aurions évidemment préféré vous proposer une solution utilisant Windows si elle existait.

Mais Linux est généralement l'environnement de développement préféré par la communauté Python, c'est pourquoi nous vous proposons une solution alternative via l'installation d'une machine virtuelle.

### Est-il possible de récupérer les PPT ?

Les fichiers powerpoint ne sont pas disponibles.
