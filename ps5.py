# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: 
# Collaborators (discussion):
# Time:

import numpy
import pylab
import re
import sklearn.metrics as metrics
import functools as ft
import math

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

def plot(est_y, x_coords, y_coords, x_label, y_label, title):
    pylab.figure()
    pylab.plot(x_coords, est_y, "r", label = "Estimated")
    pylab.plot(x_coords, y_coords, "bo", label = "Measured")
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.title(title)
    pylab.legend()
    pylab.show()

"""
End helper code
"""

def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    coeff = []
    for d in degs:        
        coeff.append(pylab.polyfit(x, y, d))
    return coeff


def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    return metrics.r2_score(y, estimated)

def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for m in models:
        est_y = pylab.polyval(m, x)
        r_s = r_squared(y, est_y)
        s_e = se_over_slope(pylab.array(x), y, est_y, m)
        plot(est_y, x, y, "Time (Years)", "Temperature (Celsius)", "Linear Best Fit for the Temperature of New York from 1961 - 2010" + "\n" + "R^2 =" + str(r_s) + "\n" + "standard error-to-slope=" + str(s_e))

def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    avg_yc = 0
    count = 0
    avg_list = pylab.array([])
    for y in years:
        avg_yc = 0
        count = 0
        for c in multi_cities:            
            list = climate.get_yearly_temp(c, y)
            total = ft.reduce((lambda x, y: x + y), list)
            count += len(list)
            avg_yc += total
        avg_yc = avg_yc/count
        avg_list = pylab.append(avg_list, avg_yc)
    return avg_list

def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    mov_avg = pylab.array([])
    for i in range(len(y)):
        if len(mov_avg) + 1 < window_length:
            avg = pylab.sum(y[0:i + 1])/(i + 1)
            mov_avg = pylab.append(mov_avg, avg)
        else:            
            avg = pylab.sum(y[i - window_length + 1:i + 1])/window_length
            mov_avg = pylab.append(mov_avg, avg)
    return mov_avg

def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    n = float(len(y))
    sum = 0.0
    dig = 0.0
    for i in range(len(y)):
        dif =  float((estimated[i] - y[i])**2)
        sum += float(dif)
    sum = float(math.sqrt(sum/n))
    return sum

def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    std_list = pylab.array([])
    avg_yc = 0
    months = range(1,13)
    days = range(1,31)
    daily_temp = pylab.array([])
    daily_temp_sum = 0
    daily_avg = 0
    daily_avg_list = pylab.array([])
    cities_avg = gen_cities_avg(climate, multi_cities, years)
    agg_dif = 0

    for y in range(len(years)):
        for c in multi_cities:
            daily_temp = pylab.append(daily_temp, climate.get_yearly_temp(c, y))
        for i in range(len(daily_temp)):
            for j in range(len(daily_temp[i])):
                daily_temp_sum += 
        daily_avg = daily_temp_sum/len(multi_cities)
        daily_avg_list = pylab.append(daily_avg_list, daily_avg)
        daily_temp_sum = 0
        for i in daily_avg_list:
            xd = (i - cities_avg[y])**2
            agg_dif += xd
        std_list = pylab.append(std_list, float(math.sqrt(agg_dif/len(i))))

    return std_list

        for y in range(len(years)):
        for m in months:
            for d in days:
                for c in multi_cities:
                    daily_temp = climate.get_daily_temp(c, m, d, years[y])
                    daily_temp_sum += daily_temp
                daily_avg = daily_temp_sum/len(multi_cities)
                daily_avg_list = pylab.append(daily_avg_list, daily_avg)
                daily_temp_sum = 0
        for i in daily_avg_list:
            xd = (i - cities_avg[y])**2
            agg_dif += xd
        std_list = pylab.append(std_list, float(math.sqrt(agg_dif/len(i))))

def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for m in models:
        est_y = pylab.polyval(m, x)
        rmse_val = rmse(y, est_y)
        plot(est_y, x, y, "Time (Years)", "Temperature (Celsius)", "Linear Best Fit for the Temperature of New York from 1961 - 2010" + "\n" + "RMSE =" + str(rmse_val))

if __name__ == '__main__':

    # pass 

    climate = Climate("data.csv")
    x = pylab.array(TRAINING_INTERVAL)
    data = pylab.array([])
    deg = [1]

    # Part A.4
    for y in TRAINING_INTERVAL:
        data = pylab.append(data, climate.get_daily_temp("NEW YORK", 1, 10, y))
    models = generate_models(x, data, deg)
    evaluate_models_on_training(x, data, models)

    data = gen_cities_avg(climate, ["NEW YORK"], TRAINING_INTERVAL)
    models = generate_models(x, data, deg)
    evaluate_models_on_training(x, data, models)

    # Part B
    data = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)
    models = generate_models(x, data, deg)
    # evaluate_models_on_training(x, data, models)

    # Part C
    window_length = 5
    m_a = moving_average(data, window_length)
    models = generate_models(x, m_a, deg)
    # evaluate_models_on_training(x, m_a, models)

    # Part D.2
    deg = [1, 2, 20]
    models = generate_models(x, m_a, deg)
    # evaluate_models_on_training(x, m_a, models)

    # Part E
    x = pylab.array(TESTING_INTERVAL)
    data = gen_cities_avg(climate, CITIES, TESTING_INTERVAL)
    m_a = moving_average(data, window_length)
    evaluate_models_on_testing(x, m_a, models)
