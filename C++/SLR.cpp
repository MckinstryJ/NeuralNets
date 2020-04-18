#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

/*
 * Possible areas of improvement:
 *      Incorporate pointers
 *      Add Iterative method
 *      Minimize the Data Pull
 */

std::vector<float> readRecordFromFile(std::string file_name);
void SLR_Least(std::vector<float> closing);

int main() {
    std::vector<float> closing_price = readRecordFromFile("./GOOG.csv");

    SLR_Least(closing_price);

    return 0;
}

//Read data from csv file
std::vector<float> readRecordFromFile(std::string file_name) {
    /*
     * Getting all the data
     */
    std::vector<std::vector<std::string>> company_data;

    std::ifstream file;
    file.open(file_name);

    std::string line, word;
    std::vector<std::string> row;
    std::vector<float> closing_price;
    while(file >> line){
        row.clear();

        getline(file, line);
        std::stringstream s(line);

        while (getline(s, word, ',')) {
            row.push_back(word);
        }
        company_data.push_back(row);

        // Only using "Close"
        for (std::vector<std::string> row_ : company_data) {
            if (row_.size() > 0){
                closing_price.push_back(std::stof(row_.at(5)));
            }
        }
    }

    return closing_price;
}

/*
 * Simple Linear Regression Model via Least Squares
 */
void SLR_Least(std::vector<float> closing){
    float num = 0;
    float den = 0;
    float x_avg = 0;
    float y_avg = 0;

    // Getting the average of X, y
    for (int i = 0; i < closing.size()-1; i++) {
        x_avg += closing.at(i);
        y_avg += closing.at(i+1);
    }
    x_avg /= closing.size();
    y_avg /= closing.size();

    // Solving for the top and bottom part of Least Squares
    for (int i = 0; i < closing.size()-1; i++) {
        num += (closing.at(i) - x_avg) * (closing.at(i+1) - y_avg);
        den += pow(closing.at(i) - x_avg, 2);
    }

    // For equation y = mx + b
    float m = num / den;
    float b = y_avg - m * x_avg;

    std::cout << "SLR is: y = (" << m << ")x + " << b << std::endl;
}
