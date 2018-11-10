{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "np.random.seed(0) # Keep it consistent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.set_style(\"whitegrid\")\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x11cabefd0>"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXMAAAD7CAYAAACYLnSTAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAGoFJREFUeJzt3X9sHGeZB/DvxjbrxSR1EC4oK0obpHu7XC2auIJC4K6N1AvQ6DApp556nHJVezpOrYBUpA0RlTipugRVqLQqlU5qKkVqAxUJNdxFpUh14Y+GSL2tjXJo+54gd420RRcXYvLDXmM7e3+46+yOZ3bnx/vOvPO+389fyWYy+45n/Mw7z/u87xSazSaIiCjf1mXdACIiSo7BnIjIAgzmREQWYDAnIrIAgzkRkQUYzImILMBgTkRkAQZzIiILMJgTEVmgP60vmp6ebhaLxbS+LrSFhQWY2C4dXDpWgMdrO1eOd25u7u2xsbGRXtulFsyLxSIqlUpaXxdarVYzsl06uHSsAI/Xdq4cb7VafTPMdkyzEBFZgMGciMgCDOZERBZgMCcisgCDORGRBRjMiYgskFppIrlnYqqOR1+SeGt2HpuGS9i7Q2B8SznrZhFZicGctJiYquMbPzqF+cVlAEB9dh7f+NEpAGBAJ9KAaRbS4tGX5Gogb5lfXMajL8mMWkRkNwZz0uKt2flInxNRMgzmpMWm4VKkz4koGQZz0mLvDoHSQF/HZ6WBPuzdITJqEZHdOABKWrQGOVnNQpQOBnPSZnxLmcGbKCVMsxARWYDBnIjIAgzmREQWYDAnIrIAgzkRkQUYzImILBCrNFEIMQDgGQDXAigCeERK+ROF7SIyCleAJNPF7Zl/CcDvpZSfBvAZAE+qaxKRWVorQNZn59HElRUgJ6bqWTeNaFXcYP5DAA+/8+cCgCU1zSEyD1eApDyIlWaRUl4EACHEegBHAXxTZaOITMIVICkPYk/nF0J8EMALAJ6SUh7ptf3CwgJqtVrcr9Om0WgY2S4dXDpWQN3xjgz14+yltQ+fI0P9Rv08eX7dFncA9P0Afgbgfinly2H+T7FYRKVSifN1WtVqNSPbpYNLxwqoO979Ozd0vDUJWFkBcv/OG1CpmDMIyvNrp2q1Gmq7uD3z/QA2AnhYCNHKnX9WSsnnTrIOV4CkPIibM/8qgK8qbguRsbgCJJmOk4aIiCzAYE5EZAG+nILIB2d8Ut4wmBN5tGZ8tqpXWjM+ATCge/CmZw6mWYg8OOMzHC5zYBYGcyIPzvgMhzc9szCYE3lsGi5F+txVvOmZhcGcyGPvDoHSQF/HZ6WBPuzdITJqkZl40zMLgzmRx/iWMg7sGkV5uIQCgPJwCQd2jXJgz4M3PbOwmoXIB2d89sZlDszCYE7GYtmb+XjTMweDORmJtd5E0TBnTkZi2RtRNAzmZCSWvRFFwzQLGWnTcAl1n8DNsjeOJZA/BnMy0t4dwvftPi6XvU1M1fGtn/was/OLq5+1jyWIwaxaRiZgmoWMxFrvTq0B4fZA3sKxBALYMyeDseztCr8B4XYcSyAG8xzLKnfKnG36egVrv7EEnie3MJjnVFZ12Kz/zkbQgDDQPpZwfvUznif3MGeeU1nVYbP+Oxt+66AAwMZ3D/iOJfA8uYc985zKqg6b9d/ZiLoOCs+TexjMcyqrOmzWf2cnyoAwz5N7mGbJqayWH+Wyp/nA8+Qe9sxzKqvlR01Y9pRVGr2ZcJ4oXQzmOZZVHXaW9d+s0giPdfpuYZqFcoVVGkT+2DOnXDG5SoPpH8pSomAuhPg4gG9LKW9R0xyi7kyt0lCV/uENgeKKnWYRQjwI4GkAXKuNUmNqlYaK9E/rhlCfnUcTV24IE1N1xa3N1sRUHdsOTuK6fcex7eCkdceXlSQ5898C2KWqIURhmLqaoor0jwvjAa7csLIQO80ipTwmhLhWYVuIQjGxSkNF+sfk8QBVut2wTDuneZPaAOjCwgJqtVpaXxdao9Ewsl06uHSsQLrHe9foe/DEiQYWlpurnxX7Crhr9D2h2zAy1I+zl5Z8Pw+zjzyc3243rKhtz8Pxpim1YF4sFlGpVNL6utBqtZqR7dLBpWMF0j3eSgUob0o2eLl/5wbftyvt33kDKpXe+8nD+d00/LvAJ5iobc/D8apQrVZDbWd0aSJH9ilPkqZ/XJi1ydcB6pMomEsp/xfAzWqa0okz/ZLxuxHyHZHmM3E8QCUXblhZMbZn7sJAiaonD+9+br1+BMeq9TU3wvtvfi8ceColw9l+w8qKscHc9pF9lZNMvPt57uQZND3bzS8u4/Dr53Df7UqanxtM1ZErjF2bJaikK+uZfqqoqin22483kLfM+FRK2Gzy9AXWNJMzjA3mps70U0XVk0eU7UeGjH0Q0+Lw6+esn4RD1GLsb7ftAyWq1hgJ2k8BnT300kAfdm/dGLGV+Rb0JGJLqi6vmPrSw9hgDtg9UKKqRCtoP3eMlfHKGzOeapbzXfZkn6BJOLak6vKIVWr6GB3MbabqySPKfmo1t4L57q0b8eTJPyiraWaPMjkXqtSywmCeIVVPHjY/wSSxffN6lDeVlZV/skeZnO1ValliMCerqbrRmdSjzPMTgqnr0duAwZwoBFN6lN2eELKY4Rv1xsLp/PoYW5pIlMTEVB27j55R9gIEU+Y9mLTmeZy1yU1dj94G7JmTdXTkt03pUcZ5QtCVlombeuIYjx7smZN1dPReTelRRn1C0PlmH1NST7SCPXMyWpxepa4gE6VHqas33P0JYW3pqc6BWw5mmiXXPXO+GNZucXuVWee3dfaGoz4h6Ow9277kRt7ktmfOul/7xe1V6shvR+lp6y5jjPKEoLP3bPuSG3mT22CeRd1vnut78yhur7J1Tv71P/4LM5eWEp+rqB2HuO3WcX3pHrjlYKY5chvM0x58seFJYPL0Bdz748nc3IyS9CrHt5QhBs8reUdk1I5DnHbrur7Ye3ZHbnPmaedFTarvjWNiqo4nTrydq7W9TcnJRu04xGm3zutrfEsZr+7bjv85eDte3bedgdxSuQ3maf+i570M69GXJBaWO19bofpmpHpAOi/lgN7jBhC53d2uLw70Uxi5S7O05xWvKg1gcGAdZucWtT8+5r0MS/fNSGeaIOueZLe8c9BxH9g1ilf3bQ/9HUHX11Wlgdyn9ygdueqZe0u+ZucX0Vi8jMfuvFH746Mpj/xx6U5LmZCG8vZgJ09fULLfbk8Iqo476PoqFJD5z5XyIVc98yxXrsv7QNLeHQIPHf1VR6olys3IW2lx6/UjHS+/8OtVAp09fx3VGq191mfnO96uVJ+dxxMnGihvqmstB1T1xBN0fe15flrJ/sl+uQrmWeetTXjkj2t8Sxn1t+o4cupi5GDql0p49uSZ1X/3BtJ27Xll1ekC7z6937+w3NR+o1eZfvO7vlo3KhX7J7vlKpjHLfnKa29ate2b1+O+2z8W+f/5PRF5NeH/3tFWz1/HU1WYdqm+0fs9oRyr1rXVcZuywBeZL1c586h5a53Tql0SNiA2gcAKDtVPVRNT9cDUTrteN/ooVSJ+19Oxah13jJW1VdyYUtFD5stVzzxq3tqkt8PkWbeceLvycCmwgkNlOqIVVHsp9hV63uijpH2CrqdX3piJVLnSi9/TpMr9d/se/l7kV66CORAtb511jt0Wfo/6Xr0e/VWmC7qlV1qpnvJwCXeNvkfpjT6N68nvJrPn+Wn855t/wCPjo1q/hyWP+WZ8ME/Se8h7bbgp/J6IvNUsvc6LymqgbsHzsTtvXN1nrVaLvI9u+07jevK7yTSB1QFnVQGdT632iR3MhRDrADwF4KMAFgDcK6X8jaqGAcl7Dxw8UkdFJY+qaqCgoFoeLmm90adxPXW7mTx38gxu+tB7lfwM+dRqnyQDoOMABqWUnwCwD8B31DTpiqQTMjh4ZCcVE7j89gEAlxaWAgdCx7eUccdYGX2FAgCgr1DAHWNqy1W73UyagLLJQlmv+Z6U6ne82iBJmuVTAH4KAFLKk0KIm9Q06QoVvYc814ar0J6mGhnqx/6dG3L/81CRsmlt+y///mucm1tc/Xx2fjHw6W9iqo5j1TqWmysFmMvNJo5V68p6ywBWJwr51ewD6nrOeX5qzUu+P+0B5kKzGXTZdCeEeBrAMSnli+/8/QyAzVLKJb/tp6enm8ViMdJ37D56Bmcvrd3d1UP9OPzFa6I32kej0cDg4KCSfZlm8vQFPHHi7Y5Zn8W+Ar7yyfdh++b1GbYsHWHObZRrLI3rEQCePDmD49J/KYJu3xX1Wp48fQGHXz+HmUtLGBnqx+6tG3NxXaR1HpJQ+bs3NzdXHRsb69lZTtIzPw+gvVXrggI5ABSLxchrS+/fucG397B/5w2oVNTc4Wq1mpI1r010748n16yUuLDcxJFTF3tOHjKpbC1uW8Kc25lLpwM+X1rzf6Nsm8T3KhVsnDiF506e6eihFwCcvbSEe3/8O9+fQdRruVIB7rtdTZvTlNZ5SCLJ755XtVoNtV2SnPmrAD4HAEKImwH0LvyNiDnvZJK88caUyVa62xIld5xmnvmR8VE8dueNKL+zb++6My5PfstDvj+LAeYkwfwFAA0hxAkAjwHYo6ZJnbiwfnzdLvpusx9NWgHxa89Pa21LlMHUtFfObF375eHSmhy6yysn5mEF0yxuOLHTLFLKywC+rLAtpFjQINet1490HUDSMfU+SprEO8Clsi1eUQZTs1o5k2WEnVS/41WHLAaYjZ80RPF5g89KNcsNPSeM6Jh6H6byoH05215U9nCiVDxlUR0V93yYNO6hmsp3vOqQxY2fwdxy7cFnZYCs3HONbN1T7/1mGobpjSdtS175nY+BdQXM/WkJ1+07vhooRFshS17K92yW9o0/V6smkhq98nkqB57DpgjCLGeLhG3JK+/5GC4NAAXg3Nxix6Bw+5uVTBj3oHSxZ+6gMD1v3VPvvTeUXvnf0kCfc0G8Xfv52HZwErPzix3/Pr+4jMOvn1stNWSe3T0M5pbrNgM0jXxe2JRNt2V2y5ble5MKCsgzbRNpuMicexjMLebNm569tNSRN00jOIa9cQQFfRt646oHIoMC9cjQlV/nPE/Xp3gYzC1myjKnYW4ceX9hdhAdA5FBgXr31o2rf7f150nBGMwt1itvalrpmo2Loum4oQYFajF4fs12tv08KRiDucW65U2zLl0z7Uaii66BSL9AXaudD9iaXMDSRIt1m/asq3QtzEuSTVr7Rbc8rCNCdmAwt5i3Pvnqof7VAcWkPUa/oB02SLtUA52HdUTIDk6mWWx8YUMQvxmgQLLStaAUzeDAulD5YZdqoDkQSWlxLpj3KtdzRZLStaCeddAMTm+Qdq0GmgORlAbn0ix5fMQPk4eOKsmU/ag9aG+Q9ks9FLDSw+f7HInica5nnrdHfJ1VJ3F7jEE96+HSABaWLvfs7benHuqz874vXmjfjtRxpYrIRc71zIMe5ZuAkb1CE58kggb1vvXXfx66t88XL6TPpSoiFznXM/fLFbeY2Cs08Umi16BelJ+dicdnK1NmBJMezgVz7yO+l2kXt6mDhbpXVVxXKHSs1W3K+cgz3jjt5lyaBbjyiF8I+HeTLm7b65T9jg8AlptNpgIU4wQmuzkZzFtMvbjbq1cefUnijrGykhdFmMhbVdNXWHuLZQ5dDds7Bq5zLs3Sbu8OgYeO/goLy1eG4LK+uP2qV45V68oCuInVDO0pm+v2HffdxqSnpbziBCa7OR3Mx7eUUX+rjiOnLhpzcescpJo8fQFPnnzT6PdCBpY9vnsA2w5OGnOe8ooTmOzldDAHgO2b1+O+2z+WdTNW6RykOvz6OeOrGXxfXtxXwMXGEs7NrbwqzcSbEFHWnM6Zm0hnHr/9tWLtTEph+M1MHXpXPxYvd1ajM4+uh47ZxpQO53vmKqjMQ+t83dfIUD/O+gT0rAd8vbypgDTz6CaOKaQl6zXuKRn2zBNSPasuyZopvezeujF31QwTU3Ws86lwAdTfhFyfIWnibGMKjz3zhHS9FkxHT2j75vUobyrnpufZCq7LTe+Efz03objn0pbePCcV5RuDeUJ5+wVIq5pBRYDzC67ASi26jlr7OOfSptSEqbONKZxEaRYhxBeEEEdUNSaPTJ14lCVV6YqgIHq52dQSKOOcS5tSE5xUlG+xg7kQ4nEAB5Lswwb8BVhLVYBL+0YZ51zm7cmsG53jNaRfkjTLCQATAP5JUVtyibPq1lIV4HRW9viJcy5tS01wUlF+FZo+g0vthBD3ANjj+fhuKeVrQohbAHxZSvm3vb5oenq6WSwWYzdUl0ajgcHBwaybkYq0jnX30TO+JZBXD/Xj8BevibSvJ0/O4MX/voDLTWBdAfjsn63H/TePhPq/aRzv5OkLeOLE2x1LQhT7CvjKJ9+H7ZvXa/1uL5euZcCd452bm6uOjY3d1Gu7nj1zKeUhAIeSNqhYLKJSqSTdjXIrLzk2r106hD3WqIOX3u3/anQTjlXra3rU+3fesPpC6TAmpuqYPP0mWvOFLjeBydNzuO3GcC/gTuPcVipAeZMZ1SwuXcuAO8dbrVZDbcdqFuoQtTojaGGwO8bKeOWNmcAAF+aGkZeXKTA1QSZgMKcOUQNo0PavvDGDV/dt9/2OsDcMmwYXiXRLFMyllD8H8HMlLSEjRA2gcQJu2BtGmoOLtkz8IXc5XVZIa0UtB4xTPhj2BpBW2afr0/jJDgzm1CFqAI0TcMPeANKqe7Zp4g+5izlz6hC11jpObXaU+vE0BheZmycbMJjTGlEDaJztAXMmWtk28YfcxGBOmTCpnC/tmaZEOjCYk/NMe1IgioPBnAhmPSkQxcFgTtqxhptIPwZz0sqmlzcQmYx15qQVa7iJ0sFgTlqxhpsoHQzmpBVfq0eUDgZzxSam6th2cBLX7TuObQcnnV/fg6/VI0oHB0AV4mDfWqzhJkoHg7lCeXmZQtpYw02kH9MsCnGwj4iywmCuEAf7iCgrDOYKcbCPiLLCnLlCHOwjoqwwmCvGwT4iygKDOTmPC4GRDRjMyWmcG0C2YDAnpyWdG8BePZmCwZyclmRuAHv1ZBKWJpLTkswN4PK+ZBIGc3JakrkBnPFLJmEwJ6eNbynjwK5RlIdLKAAoD5dwYNdoqDQJZ/ySSZgzNxQH1tITd27A3h2iI2cOcMYvZSdWMBdCXAXgWQAbALwLwANSyl+qbJjLOLCWD5zxSyaJ2zN/AMDLUsrvCiEEgO8D2KquWW7jUrr5wRm/ZIq4wfwxAAtt+2ioaQ4BHFgjouh6BnMhxD0A9ng+vltK+ZoQ4gNYSbd8rdd+FhYWUKvV4rVSo0ajYVy7Rob6cfbSku/nSdpq4rHqxOO1m2vH20vPYC6lPATgkPdzIcQogB8A+LqU8he99lMsFlGpVGI1UqdarWZcu/bv3OA7sLZ/5w2oVOI/0pt4rDrxeO3myvFWq9VQ28UdAP0IgB8CuFNK+as4+6BgHFgjoqji5swPABgE8PjK+Cf+KKX8vLJWEQfWiCiSWMGcgZuiYM08kX6cNERasWaeKB2czk9acTEqonQwmJNWrJknSgeDOWnFxaiI0sFgTlolWWKWiMLjAChpxZp5onQwmJN2rJkn0o9pFiIiCzCYExFZgMGciMgCDOZERBZgMCcisgCrWYgsxMXN3MNgTmQZLm7mJqZZiCzDxc3cxJ45keGipky4uJmb2DMnMlgrZVKfnUcTV1ImE1P1wP/Dxc3cxGBOZLA4KRMubuYmplmIDBaUGqnPzmPbwcmO1IsYXPk3Lm7mJgZzIoNtGi6h7hPQC8Dq563Uy/03vxeVysq/c3Ez9zDNQmQwv5RJAUDTs9384jIOv34utXaReRjMiQw2vqWMA7tGUR4uoQCgPFxaE8hbZi4tpdk0MgzTLESG86ZMth2c9E29jAzx19ll7JkT5UxQtcrurRszahGZgLdyopwJqlYRg+czbhllicGcKIf8qlVqNQZzlzHNQkRkAQZzIiILxEqzCCGGABwBsBHAnwDsllIGLxZBRERaxe2Z/yOAqpTyLwA8C+BBdU0iF0xM1bHt4CSu23cc2w5Odl04ioh6i9Uzl1J+VwjRqo26BsCsuiaR7fjyBCL1Cs1m0HyyFUKIewDs8Xx8t5TyNSHEJIBRALdJKae77Wd6erpZLBYTNVaHRqOBwcHBrJuRClOOdffRMzjrM1vx6qF+HP7iNcq+x5TjTQuP105zc3PVsbGxm3pt17NnLqU8BOBQwL9tF0JcD+A4gA9320+xWESltQqQQWq1mpHt0sGUY525dDrg8yWl7TPleNPC47VTtVoNtV2snLkQ4htCiL9/568XASx3256oHV+eQKRe3AHQZwD8nRDi5wC+D+BuZS0i6/HlCUTqxR0A/T8An1HcFnIEX55ApB6n81Mm+PIEIrU4A5SIyAIM5kREFmAwJyKyAIM5EZEFGMyJiCzAYE5EZIGea7OoUq1WZwC8mcqXERHZ40NjY2MjvTZKLZgTEZE+TLMQEVmAwZyIyAIM5kREFmAwJyKyAIM5EZEFnF41UQhxFVZeSL0BwLsAPCCl/GW2rdJPCPEFAH8jpbwr67boIIRYB+ApAB8FsADgXinlb7JtlX5CiI8D+LaU8pas26KTEGIAK+9UuBZAEcAjUsqfZNooA7jeM38AwMtSyr8E8A8Avpdtc/QTQjwO4ADsPvfjAAallJ8AsA/AdzJuj3ZCiAcBPA3A/pdiAl8C8Hsp5aex8l6FJzNujxFs/oUO4zEA//bOn/sBNDJsS1pOAPjnrBuh2acA/BQApJQnAfR8Ga4FfgtgV9aNSMkPATz8zp8LANa+HdxBzqRZhBD3ANjj+fhuKeVrQogPYCXd8rX0W6ZHl+N9XghxSwZNStMGAH9s+/uyEKJfSmntL72U8pgQ4tqs25EGKeVFABBCrAdwFMA3s22RGZwJ5lLKQwAOeT8XQowC+AGAr0spf5F6wzQJOl5HnAewvu3v62wO5C4SQnwQwAsAnpJSHsm6PSZwOs0ihPgIVh7Z7pJSvph1e0iZVwF8DgCEEDcDOJVtc0glIcT7AfwMwENSymeybo8pnOmZBziAlQGjx4UQAPBHKeXns20SKfACgNuEECewklO9O+P2kFr7AWwE8LAQopU7/6yUcj7DNmWOC20REVnA6TQLEZEtGMyJiCzAYE5EZAEGcyIiCzCYExFZgMGciMgCDOZERBZgMCcissD/A4iE/We5PIV+AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Get random data \n",
    "x_inital = np.random.randn(100,2)\n",
    "plt.scatter(x_inital[:,0], x_inital[:,1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get a new point ! \n",
    "x_new = np.random.randn(1,2)\n",
    "# Create the updated data set \n",
    "x = np.zeros((101,2))\n",
    "x[0:100,:] = x_inital\n",
    "x[100:,:] = x_new"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x11cb295f8>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot the updated data set\n",
    "plt.scatter(x[:,0], x[:,1], c=\"#43c0ac\")\n",
    "plt.scatter(x_inital[:,0], x_inital[:,1], c=\"#ff7657\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x11d36ff98>"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Generate the bounding box\n",
    "# Note that the box is first generated based on the inital data\n",
    "B_s = np.zeros((x.shape[1],2))# [min, max] of dx2\n",
    "B_s[:,0] = np.min(x_inital, axis=0) # min of column x_inital\n",
    "B_s[:,1] = np.max(x_inital, axis=0) # max of column x_inital \n",
    "plt.axvspan(B_s[0,0], B_s[0,1], facecolor='g', alpha=0.25)\n",
    "plt.axhspan(B_s[1,0], B_s[1,1], facecolor='b', alpha=0.25)\n",
    "plt.scatter(x_inital[:,0], x_inital[:,1], c=\"#ff7657\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Update the bounding box based on the internal point\n",
    "# This is stupid ! why not just do it in the above.\n",
    "B_s[:,0] = np.minimum(B_s[:,0], x_new) # compare element wise \n",
    "B_s[:,1] = np.maximum(B_s[:,1], x_new)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2.6311019230334645\n"
     ]
    }
   ],
   "source": [
    "# Find r \n",
    "temp_sum = sum(B_s[:,1]-B_s[:,0])\n",
    "r = np.random.choice(np.linspace(0, temp_sum, int(temp_sum*10)), 1)[0]\n",
    "print(r)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Identify the cut.\n",
    "temp_diff = B_s[:,1]-B_s[:,0]\n",
    "obj = np.cumsum(temp_diff)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    "j = 9999\n",
    "for i in range(0,len(obj)):\n",
    "    if obj[i] >= r:\n",
    "        j = i\n",
    "        break"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [],
   "source": [
    "cut = B_s[j,0] + obj[j] - r"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.06512212953011476"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cut"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "def InsertPoint_cut(S ,p):\n",
    "    \"\"\"\n",
    "    Generates the cut dimension and cut value \n",
    "    based on the Insertpoint algorithm \n",
    "    ---- \n",
    "    Inputs:\n",
    "    S : Set of point to be split (numpy array (n x d))\n",
    "    p : New point to be inserted (numpy array (1 x d))\n",
    "    \n",
    "    Returs:\n",
    "    dimenstion for cut, cut value \n",
    "    ----\n",
    "    Example:\n",
    "    InsertPoint_cut(x_inital, x_new)\n",
    "    (0, 0.9758881798109296)\n",
    "    \"\"\"\n",
    "    # Generate the bounding box\n",
    "    \n",
    "    # Note that the box is first generated based on the inital data\n",
    "    B_s = np.zeros((S.shape[1],2))# [min, max] of dx2\n",
    "    B_s[:,0] = np.min(S, axis=0) # min of column x_inital\n",
    "    B_s[:,1] = np.max(S, axis=0) # max of column x_inital \n",
    "    \n",
    "    # Update the bounding box based on the internal point\n",
    "    \n",
    "    # This is stupid ! why not just do it in the above.\n",
    "    B_s[:,0] = np.minimum(B_s[:,0], p) # compare element wise \n",
    "    B_s[:,1] = np.maximum(B_s[:,1], p)\n",
    "    \n",
    "    # Find r \n",
    "    temp_sum = sum(B_s[:,1]-B_s[:,0])\n",
    "    # resolution of r increases as the r magnitude increases, 10 times for now.\n",
    "    temp_len = np.linspace(0, temp_sum, int(temp_sum*10)) \n",
    "    r = np.random.choice(temp_len, 1)[0] # returns a float.\n",
    "    \n",
    "    # Identify the cut.\n",
    "    temp_diff = B_s[:,1]-B_s[:,0]\n",
    "    obj = np.cumsum(temp_diff)\n",
    "    \n",
    "    cut_dimenstion = 9999 # Hope we do not have 999 dimentional data\n",
    "    for i in range(0,len(obj)):\n",
    "        if obj[i] >= r:\n",
    "            cut_dimenstion = i\n",
    "            break\n",
    "            \n",
    "    cut = B_s[j,0] + obj[j] - r\n",
    "    \n",
    "    return cut_dimenstion, cut"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0, 0.9758881798109296)"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "InsertPoint_cut(x_inital, x_new)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
