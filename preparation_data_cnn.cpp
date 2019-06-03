#include <stdio.h>
#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
#include <map>

#include <opencv2/opencv.hpp>


//using namespace cv;
//using namespace std;

int main(int argc, char** argv)
{
	srand(time(0));
	
	int programme_choisi;
	std::cout << "1 : Préparation des données : resize des images, images enregistrer en grayscale et sauvegarde des chemins d'accès et des labels dans un fichier texte" << std::endl;
	std::cout << "2 : Séparation des chemins d'accès et des labels dans deux fichiers textes" << std::endl;
	std::cout << "3 : Mise en forme" << std::endl;
	
	std::cin >> programme_choisi;

	/****************************************************************************************************************************************/
	/*                                 Apprendre à utiliser la fonction glob pour lire les fichiers d'un dossier                            */
	/****************************************************************************************************************************************/

	if (programme_choisi == 1)
	{


			/*                                   Lecture automatique des tous les dossiers en précisant en avances leurs noms                     */

		cv::String Path = ("../preparation_data/base_entrainement/");
		std::vector<cv::String> names = { "Classe_1", "Classe_2", "Classe_3", "Classe_4", "Classe_5" };

		cv::String image_folder;
		cv::String prep_data_file;
		std::ofstream myfile_synthese;
		myfile_synthese.open(Path + "synthese.txt");
		for (int j = 0; j < names.size(); j++)
		{
			image_folder = Path + names[j] + "/*.jpg";
			prep_data_file = Path + "prep_" + names[j] + "/";

			/*                                    Lecture des chemins d'accès aux images et stockage dans la variable filenames                       */

			// Return the image filenames inside the image folder
			std::vector<cv::String> filenames;
			cv::String folder(image_folder);
			glob(folder, filenames);

			// Loop through each image stored in the images folder
			// Each images are convert to grayscale and resize
			// At the end all the news file name and label and write in a .txt file
			int x_width = 28;
			int y_height = 28;

			//  ofstream: Stream class to write on files/ ifstream : Stream class to read from files/ fstream : Stream class to both read and write from / to files
			std::ofstream myfile;
			myfile.open(prep_data_file + names[j] + ".txt");
			

			if (!myfile)
			{
				std::cerr << "ERREUR à l'ouverture de " << prep_data_file + names[j] + ".txt" << std::endl;
			}

			for (int i = 0; i < filenames.size(); i++)
			{
				// Read in an image
				cv::Mat sample_image = cv::imread(filenames[i], cv::IMREAD_GRAYSCALE);

				// Check if the image is actually read - avoid other files in the folder, because glob() takes them all
				// If not then simply skip this iteration
				if (sample_image.empty()) // Check for failure
				{
					continue;
				}

				cv::String  temp_prep_data_file = prep_data_file + std::to_string(i) + ".jpg";

				resize(sample_image, sample_image, cv::Size(x_width, y_height), 0, 0, cv::INTER_LINEAR_EXACT);
				cv::imwrite(temp_prep_data_file, sample_image);
				myfile << temp_prep_data_file + " " + std::to_string(j) + "\n";
				myfile_synthese << temp_prep_data_file + " " + std::to_string(j) + "\n";
				
			}
			myfile.close();

		}
		myfile_synthese.close();

		
	}
	/****************************************************************************************************************************************/
	/*                         Lecture du fichier texte pour ouvrir et utiliser les images et séparer les labels                            */
	/****************************************************************************************************************************************/
	if (programme_choisi == 2)
	{
		cv::String Path = ("../preparation_data/base_entrainement/");
		std::ifstream myfile(Path + "synthese.txt");
		
		std::ofstream label_file;
		label_file.open(Path + "label.txt");
		std::ofstream imagepath_file;
		imagepath_file.open(Path + "imagefile.txt");

		std::string adresse;
		int label;
		

			if (myfile)  // si l'ouverture a fonctionné
			{
				std::string ligne;

				do  // tant que l'on peut mettre la ligne dans "contenu"
				{
					
					myfile >> adresse >> label;
					
					if (myfile.eof() == false)
					{
						label_file << label << "\n";
						imagepath_file << adresse << "\n";
						std::cout << adresse << std::endl;
						
					}
				} while (std::getline(myfile, ligne));
				
			}
			else
				std::cerr << "Impossible d'ouvrir le fichier !" << std::endl;

		
			myfile.close();
	
	}

	/****************************************************************************************************************************************/
	/*                                                Mise en forme des différentes données                                                 */
	/****************************************************************************************************************************************/

	if (programme_choisi == 3)
	{
		cv::String Path = ("../preparation_data/base_entrainement/");
		std::ifstream label_file;
		label_file.open(Path + "label.txt");

		std::ifstream imagepath_file;
		imagepath_file.open(Path + "imagefile.txt");

		int temp_label;
		std::string adresse;
		std::string ligne;
		std::vector<cv::String> filenames;
		std::vector<int> label;
		std::vector<cv::Mat> image_stockage;
		int count = 0;
		/*                                Extraction des chemins d'accès et des labels des différents fichiers textes                       */

		if (imagepath_file && label_file)  // si l'ouverture a fonctionné
		{
			std::string ligne;

			do
			{
				imagepath_file >> adresse;
				label_file >> temp_label;

				if (imagepath_file.eof() == false && label_file.eof() == false)
				{
					filenames.push_back(adresse); // non nécessaire si l'on veur uniquement remplir le tableau d'image et ne pas stocké les chemin d'accès
					
					cv::Mat temp_image = cv::imread(adresse);

					label.push_back(temp_label);
					image_stockage.push_back(temp_image);   /********** ESSAYER LA FONCTION IMREADMULTI****************************/

					count++;
				}
			} while (std::getline(imagepath_file, ligne));

		}
		else
			std::cerr << "Impossible d'ouvrir un des fichiers !" << std::endl;


		imagepath_file.close();
		label_file.close();

		/*                                                  Mise en forme des image et des labels                                          */

		if (image_stockage.size() != label.size())
		{
			std::cout << "Une erreur a eu lieu quelque part lors de la mise en forme des données" << std::endl;
		}

		int const nb_label = 5; // on connait ce nombre normalement il n'y a besoin de détecter automatiquement ce nb

		std::vector<int> one_hot_temp(nb_label);
		std::vector<std::vector<int>> one_hot_label;

		for (int i = 0; i < nb_label; i++)

		{
			one_hot_temp[i] = 1;
			one_hot_label.push_back(one_hot_temp);
			one_hot_temp[i] = 0;
		}
		std::ofstream one_hot_file;
		one_hot_file.open(Path + "one_hot.txt");

		std::vector<std::vector<int>> one_hot_label_image;
		for (int k = 0; k < count; k++)
		{
			if (label[k] == 0)
			{
				one_hot_temp[0] = 1;
				one_hot_label_image.push_back(one_hot_temp);
				one_hot_temp[0] = 0;
			}
			else if (label[k] == 1)
			{
				one_hot_temp[1] = 1;
				one_hot_label_image.push_back(one_hot_temp);
				one_hot_temp[1] = 0;
			}
			else if (label[k] == 2)
			{
				one_hot_temp[2] = 1;
				one_hot_label_image.push_back(one_hot_temp);
				one_hot_temp[2] = 0;
			}
			else if (label[k] == 3)
			{
				one_hot_temp[3] = 1;
				one_hot_label_image.push_back(one_hot_temp);
				one_hot_temp[3] = 0;
			}
			else if (label[k] == 4)
			{
				one_hot_temp[4] = 1;
				one_hot_label_image.push_back(one_hot_temp);
				one_hot_temp[4] = 0;
			}
			

		}
		
				

		/*                                                  Mise en forme des image                                                          */
		cv::Mat image_stockage_ligne = cv::Mat::zeros(count, 900, CV_8UC1);
		cv::Mat image_stockage_temp = cv::Mat::zeros(20,20,CV_8UC1);

		for (int k = 0; k < count; k++)
		{
			image_stockage_temp = image_stockage[k];

			for (int i = 0; i < 20; i++)
			{
				for (int j = 0; j < 20; j++)

				{
					image_stockage_ligne.at<unsigned char>(k,j + i * 20) = image_stockage_temp.at<unsigned char>(i, j);
				}
			}
		}
		image_stockage_ligne.convertTo(image_stockage_ligne, CV_32F, 1.0 / 255); // conversion du cv::mat 8 bit non signé 1 channel en cv::mat vers flottant entre 0 et 1

		cv::String windowName_file_hh = "Original image jyjy"; //Name of the window
		namedWindow(windowName_file_hh); // Create a window
		imshow(windowName_file_hh, image_stockage_ligne); // Show our image inside the created window.

	}

