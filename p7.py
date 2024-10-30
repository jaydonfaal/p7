#Section 1

import numpy as np
import matplotlib.pyplot as plt


#Section 2
class Growth_model:
    def __init__(self, para_dict, state_dict):
        # Parameter and state dictionaries
        self.para_dict = para_dict
        self.state_dict = state_dict

        # Calculate initial state variables
        self.state_dict['y'] = self.para_dict['a'][0] * self.state_dict['k']**self.para_dict['alpha'][0]
        self.state_dict['K'] = self.state_dict['k'] * self.state_dict['L']
        self.state_dict['Y'] = self.para_dict['a'][0] * self.state_dict['K']**self.para_dict['alpha'][0] * self.state_dict['L']**(1 - self.para_dict['alpha'][0])
        self.state_dict['i'] = self.para_dict['s'][0] * self.state_dict['y']
        self.state_dict['I'] = self.state_dict['i'] * self.state_dict['L']
        self.steady_state = {}

    def growth(self, years):
        # Define the timeline
        time_line = np.linspace(0, years, num=years+1, dtype=int)
        
        for t in time_line: 
            n = self.para_dict.get('n')[0]
            s = self.para_dict.get('s')[0]
            alpha = self.para_dict.get('alpha')[0]
            delta = self.para_dict.get('delta')[0]
            a = self.para_dict.get('a')[0]
            
            # Load current states
            y_t = self.state_dict['y']
            k_t = self.state_dict['k']
            Y_t = self.state_dict['Y']
            L_t = self.state_dict['L']
            K_t = self.state_dict['K']
            i_t = self.state_dict['i']
            I_t = self.state_dict['I']

            # Calculate new states
            dk = s * y_t - (delta + n) * k_t
            k_next = k_t + dk
            y_next = a * k_next**alpha
            K_next = k_next * L_t
            Y_next = a * (K_next ** alpha) * (L_t ** (1 - alpha))
            i_next = s * y_next
            I_next = i_next * L_t

            # Update state_dict
            self.state_dict['k'] = k_next
            self.state_dict['y'] = y_next
            self.state_dict['Y'] = Y_next
            self.state_dict['K'] = K_next
            self.state_dict['L'] = L_t
            self.state_dict['i'] = i_next
            self.state_dict['I'] = I_next
    
    def plot_income_growth(self, ax, years):
        time_line = np.arange(0, years + 1)
        income_data = []
        income_data.append(self.state_dict['y'])
        
        for _ in range(years):
            self.growth(1)
            income_data.append(self.state_dict['y'])
            
        ax.plot(time_line, income_data, label="Income per capita")
        ax.set_title("Income Growth Over Time")
        ax.set_xlabel("Years")
        ax.set_ylabel("Income per capita")
        ax.legend()
        
    def find_steady_state(self):
        n = self.para_dict.get('n')[0]
        s = self.para_dict.get('s')[0]
        alpha = self.para_dict.get('alpha')[0]
        delta = self.para_dict.get('delta')[0]
        a = self.para_dict.get('a')[0]
        
        k_star = (s * a / (n + delta)) ** (1 / (1 - alpha))
        y_star = a * (k_star ** alpha)
        i_star = s * y_star
        c_star = y_star - i_star
        
        steady_state = {
            "k_star": k_star,
            "y_star": y_star,
            "c_star": c_star,
            "i_star": i_star
        }

        self.steady_state = steady_state
        return steady_state

    def plot_growth(self, ax):
        
        k_values = np.linspace(0.1, 10, 100)
        s = self.para_dict.get('s')[0]
        alpha = self.para_dict.get('alpha')[0]
        a = self.para_dict.get('a')[0]
        delta = self.para_dict.get('delta')[0]
        n = self.para_dict.get('n')[0]

        y_values = a * (k_values ** alpha)
        i_values = s * y_values  
        be_values = (delta + n)*(k_values)

        ax.plot(k_values, y_values, label="Income per capita (y)", color="blue")
        ax.plot(k_values, i_values, label="Investment per capita (i)", color="green")
        ax.plot(k_values, be_values, label="Break Even Point", color="red")

        ax.set_title("Income and Investment per Capita vs. Capital per Capita")
        ax.set_xlabel("Capital per capita (k)")
        ax.set_ylabel("Per capita values")
        ax.legend()

#Section 3
parameters = {'n': np.array([0.002]), 's': np.array([0.15]), 'alpha': np.array([1/3]), 'delta': np.array([0.05]), 'a': np.array([1])}
states = {'k': np.array([1]), 'L': np.array([100])}
model = Growth_model(parameters, states)
star_values = model.find_steady_state()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
model.plot_income_growth(ax1, years=50)
model.plot_growth(ax2)
plt.tight_layout()
plt.show()

#Section 4

print(f"k star: {star_values['k_star']}\ny star: {star_values['y_star']} \nc star: {star_values['c_star']} \ni star: {star_values['i_star']}")

parameters_15 = {'n': np.array([0.002]), 's': np.array([0.15]), 'alpha': np.array([1/3]), 'delta': np.array([0.05]), 'a': np.array([1])}
model_15 = Growth_model(parameters_15, states)
print(f"c Star at 15% saving rate: {model_15.find_steady_state()['c_star']}")

parameters_33 = {'n': np.array([0.002]), 's': np.array([0.33]), 'alpha': np.array([1/3]), 'delta': np.array([0.05]), 'a': np.array([1])}
model_33 = Growth_model(parameters_33, states)
print(f"c Star at 33% saving rate: {model_33.find_steady_state()['c_star']}")

parameters_50 = {'n': np.array([0.002]), 's': np.array([0.50]), 'alpha': np.array([1/3]), 'delta': np.array([0.05]), 'a': np.array([1])}
model_50 = Growth_model(parameters_50, states)
print(f"c Star at 50% saving rate: {model_50.find_steady_state()['c_star']}")




print("\nI noticed that the c star value increases, this makes sense because as the saving rate is increasing, the steady state capital per worker increase which is going to increase the ouput per worker, thus increasing the consumption per worker as well.")
