#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

/*
 * Possible areas of improvement:
 *      Incorporate pointers
 *      Add Least Squares Method
 *      Add Genetic Algorithm Method
 *      Minimize the Data Pull
 */

std::vector<std::vector<std::string>> readRecordFromFile(std::string file_name);
void MLR_Iter(std::vector<std::vector<std::string>> cd, int epochs);

int main() {
    std::vector<std::vector<std::string>> company_data = readRecordFromFile("./GOOG.csv");

    MLR_Iter(company_data, 100);

    return 0;
}

//Read data from csv file
std::vector<std::vector<std::string>> readRecordFromFile(std::string file_name) {
    /*
     * Getting all the data
     */
    std::vector<std::vector<std::string>> company_data;

    std::ifstream file;
    file.open(file_name);

    std::string line, word;
    std::vector<std::string> row;
    while(file >> line){
        row.clear();

        getline(file, line);
        std::stringstream s(line);

        while (getline(s, word, ',')) {
            row.push_back(word);
        }
        company_data.push_back(row);
    }

    // return all data as strings
    return company_data;
}

std::vector<float> pct_change(std::vector<std::vector<std::string>> cd, int col){
    std::vector<float> data;

    // Start with 0
    data.push_back(0);

    for (std::vector<std::string> row : cd) {
        for (int i = 0; i < row.size(); i++) {
            if (i == col){
                data.push_back(std::stof(row.at(i)) / std::stof(row.at(i-1)));
            }
        }
    }

    return data;
}

float predict(float o, float h, float l, float c, float x[4]){

    return o * x[0] + h * x[1] + l * x[2] + c * x[3];
}

/*
 * Multiple Linear Regression Model via Iteration
 */
void MLR_Iter(std::vector<std::vector<std::string>> cd, int epochs){

    std::vector<float> open_pct;
    std::vector<float> high_pct;
    std::vector<float> low_pct;
    std::vector<float> close_pct;

    // Convert data to percentage change
    open_pct = pct_change(cd, 2);
    high_pct = pct_change(cd, 3);
    low_pct = pct_change(cd, 4);
    close_pct = pct_change(cd, 5);

    float x[] = {.5, .5, .5, .5};
    for (int e = 0; e < epochs; e++){
        for (int i = 1; i < open_pct.size(); i++){
            for (int j = 0; j < 4; j++){
                x[j] = 1 / x[j] * 0.05 * (predict(open_pct.at(i-1),
                                                  high_pct.at(i-1),
                                                  low_pct.at(i-1),
                                                  close_pct.at(i-1), x) - close_pct[i]);
            }
        }
    }

    std::cout << "MLR is: y = " << std::endl;
    std::cout << "     (" << x[0] << ")Open" << std::endl;
    std::cout << "   + (" << x[1] << ")High" << std::endl;
    std::cout << "   + (" << x[2] << ")Low" << std::endl;
    std::cout << "   + (" << x[3] << ")Close" << std::endl;
}
