min_y = -2
max_y = 2
min_x = -2
max_x = 2
scale = 125 #screen_width/(max-min)

def boat_transform(boat_trans, s):
    '''
    根据比例设置无人船位置与航向角
    '''
    boat_trans.set_translation(scale*(s[1]-min_y), scale*(s[0]-min_x))
    boat_trans.set_rotation(-s[2])