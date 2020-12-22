import re
import math
from collections import defaultdict
from collections import Counter
import copy
from functools import reduce


def day1():
    print('Day 1')
    array = []
    with open('2020\input1.txt') as file:
        for data in file:
            array.append(int(data.strip('\n')))

    # part 1
    for i in array:
        temp = 2020 - i
        if temp in array:
            print(i, temp)
            print(f'Part 1: {i * temp}')
            break

    # part 2
    index = 0
    done = False
    for a in array:
        for b in array[index + 1:]:
            for c in array[index + 2:]:
                if a + b + c == 2020:
                    print(a, b, c)
                    print(f'Part 2: {a * b * c}')
                    done = True
                    break
            if done:
                break
        if done:
            break
        index += 1

    print()


def day2():
    print('Day 2')
    part_1 = 0
    part_2 = 0
    with open('2020\input2.txt') as file:
        for data in file:
            counter = 0
            data = data.split(' ')
            allow_range = (int(data[0].split('-')[0]), int(data[0].split('-')[1]))
            letter = data[1][0]
            string = data[2].split('\n')[0]

            # part 1
            for i in string:
                if i == letter:
                    counter += 1

            if allow_range[0] <= counter <= allow_range[1]:
                part_1 += 1

            # part 2
            position_1 = allow_range[0] - 1
            position_2 = allow_range[1] - 1
            if string[position_1] == letter and string[position_2] != letter:
                part_2 += 1
            elif string[position_1] != letter and string[position_2] == letter:
                part_2 += 1

        print(f'Part 1: {part_1}')
        print(f'Part 2: {part_2}')
        print()


def day3Helper(array, right, down_value):
    tree_count = 0
    start = 0
    down = 0
    while True:
        try:
            if array[down][start] == '#':
                tree_count += 1
        except IndexError:
            break
        start += right
        down += down_value
    return tree_count


def day3():
    print('Day 3')
    # part 1
    array = []
    with open('2020\input3.txt') as file:
        for data in file:
            # the map will need to expand as many times as possible to get the number that will not change the answer
            # anymore
            data = data.rstrip() * 100
            array.append(data)
    part_1 = day3Helper(array, 3, 1)
    print(f'Part 1: {part_1}')

    # part 2
    first = day3Helper(array, 1, 1)
    third = day3Helper(array, 5, 1)
    forth = day3Helper(array, 7, 1)
    fifth = day3Helper(array, 1, 2)

    part_2 = part_1 * first * third * forth * fifth

    print(f'Part 2: {part_2}')


def day4Helper(valid_data):
    valid = True
    for i in valid_data:
        key = i[:3]
        if key == 'byr':
            value = i[4:]
            if len(value) != 4 or int(value) < 1920 or int(value) > 2002:
                valid = False
                break
        elif key == 'iyr':
            value = i[4:]
            if len(value) != 4 or int(value) < 2010 or int(value) > 2020:
                valid = False
                break
        elif key == 'eyr':
            value = i[4:]
            if len(value) != 4 or int(value) < 2020 or int(value) > 2030:
                valid = False
                break
        elif key == 'hgt':
            measure = i[-2:]
            digits = [a for a in i if a.isdigit()]
            digits = int(''.join(digits))
            if measure == 'cm':
                if digits < 150 or digits > 193:
                    valid = False
                    break
            elif measure == 'in':
                if digits < 59 or digits > 76:
                    valid = False
                    break
            else:
                valid = False
                break
        elif key == 'hcl':
            value = i[4:]
            pound = value[0]
            if pound != '#':
                valid = False
                break
            if len(value) > 7:
                valid = False
                break

            pattern = re.compile('[#a-f0-9]+')
            if not pattern.fullmatch(value):
                valid = False
                break
        elif key == 'ecl':
            valid_ecl = ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth']
            value = i[4:]
            if value not in valid_ecl:
                valid = False
                break
        elif key == 'pid':
            value = i[4:]
            if len(value) != 9 or not value.isdigit():
                valid = False
                break
    return valid


def day4():
    print('Day 4')
    part_1 = 0
    part_2 = 0

    # part 1
    require = {'byr': 0, 'iyr': 0, 'eyr': 0, 'hgt': 0, 'hcl': 0, 'ecl': 0,
               'pid': 0, 'cid': 0}
    # part 2
    valid_data = []
    with open('2020\input4.txt') as file:
        for data in file:
            data = data.rstrip()
            if data != '':
                data = data.split(' ')
                for i in data:
                    key = i[:3]
                    require[key] += 1
                    valid_data.append(i)
            else:
                check = sum(require.values()) - require['cid']
                if check == 7:
                    part_1 += 1
                    # part 2
                    valid = day4Helper(valid_data)
                    if valid:
                        part_2 += 1
                valid_data = []
                require = {'byr': 0, 'iyr': 0, 'eyr': 0, 'hgt': 0, 'hcl': 0, 'ecl': 0,
                           'pid': 0, 'cid': 0}

    # check the last data because there is no blank line at the end of the text
    check = sum(require.values()) - require['cid']
    if check == 7:
        part_1 += 1
        valid = day4Helper(valid_data)
        if valid:
            part_2 += 1

    print(f'Part 1: {part_1}')
    print(f'Part 2: {part_2}')


def day5Helper(data):
    row = data[:7]
    column = data[7:]

    lower_1 = 0
    upper_1 = 127

    for i in row:
        if i == 'F':
            upper_1 = math.floor((upper_1 + lower_1) / 2)
        else:
            lower_1 = math.ceil((upper_1 + lower_1) / 2)
    row = upper_1

    lower_2 = 0
    upper_2 = 7
    for i in column:
        if i == 'L':
            upper_2 = math.floor((upper_2 + lower_2) / 2)
        else:
            lower_2 = math.ceil((upper_2 + lower_2) / 2)
    column = upper_2

    return row * 8 + column


def day5():
    print('Day 5')
    # part 1
    array = []
    with open('2020\input5.txt') as file:
        for data in file:
            array.append(day5Helper(data))

    part_1 = max(array)
    print(f'Part 1: {part_1}')

    # part 2
    part_2 = 0
    array = sorted(array)
    index = 1
    limit = len(array) - 2
    while index < limit:
        seat_id = array[index]
        if seat_id - 1 not in array:
            part_2 = seat_id - 1
            break
        elif seat_id + 1 not in array:
            part_2 = seat_id + 1
            break
        index += 1

    print(array)
    print(f'Part 2: {part_2}')


def day6():
    print('Day 6')

    part_1 = 0
    part_2 = 0

    # part 1
    with open('2020/input6.txt') as file:
        answer = set()
        for data in file:
            data = data.rstrip()
            if data != '':
                for i in data:
                    answer.add(i)
            else:
                part_1 += len(answer)
                answer = set()

    # the last line in the input file
    part_1 += len(answer)
    print(f'Part 1: {part_1}')

    # part 2
    with open('2020/input6.txt') as file:
        answer = dict()
        count = 0
        for data in file:
            data = data.rstrip()
            if data != '':
                for i in data:
                    if i not in answer:
                        answer[i] = 1
                    else:
                        answer[i] += 1
                count += 1
            else:
                for i in answer:
                    if answer[i] == count:
                        part_2 += 1
                answer = dict()
                count = 0

    # the last line in the input file
    for i in answer:
        if answer[i] == count:
            part_2 += 1

    print(f'Part 2: {part_2}')


def day7HelperPart1(bag_dict, all_bags, bag_name='shiny gold'):
    bag_to_find = []
    for i in bag_dict:
        bags = bag_dict[i]
        for bag in bags:
            name, count = bag
            if name == bag_name:
                bag_to_find.append(i)

    for i in bag_to_find:
        all_bags.add(i)
        day7HelperPart1(bag_dict, all_bags, i)


def day7HelperPart2(bag_dict, all_bags, bag_name='shiny gold'):
    to_find_bag = bag_dict[bag_name]
    for i in to_find_bag:
        if i[0] != 'no other':
            all_bags.append(i[1])
            for t in range(i[1]):
                day7HelperPart2(bag_dict, all_bags, i[0])


def day7():
    print('Day 7')

    # part 1
    bag_dict = defaultdict(list)
    bag_pattern = re.compile('bag')

    with open('2020\input7.txt') as file:
        # with open('2020\\test.txt') as file:
        for data in file:
            data = data.rstrip().split(' ')

            key_bag = ''
            temp_bag = ''
            check_key_bag = False
            temp_value = 0
            for i in data:
                if bag_pattern.match(i):
                    if len(temp_bag):
                        bag_dict[key_bag].append((temp_bag, temp_value))
                        temp_bag = ''
                        temp_value = 0
                    continue

                if i.isdigit():
                    temp_value = int(i)
                elif i != 'contain':
                    if not check_key_bag:
                        if not len(key_bag):
                            key_bag += i + ' '
                        else:
                            key_bag += i
                    else:
                        if not len(temp_bag):
                            temp_bag += i + ' '
                        else:
                            temp_bag += i
                elif i == 'contain':
                    bag_dict[key_bag] = []
                    check_key_bag = True

    all_bags = set()
    day7HelperPart1(bag_dict, all_bags)
    print(f'Part 1: {len(all_bags)}')

    # part 2
    all_bags = list()
    day7HelperPart2(bag_dict, all_bags)
    print(f'Part 2: {sum(all_bags)}')


def day8Helper(instructions, corrupted_instruction_index):
    acc = 0
    not_solved = True

    for i in corrupted_instruction_index:
        if not_solved:
            if instructions[i][0] == 'nop':
                instructions[i][0] = 'jmp'
            else:
                instructions[i][0] = 'nop'

            index = 0
            acc = 0
            not_loop = False
            while True:
                try:
                    if instructions[index][2] == '2':
                        break
                except IndexError:
                    not_loop = True
                    break

                instructions[index][2] = str(int(instructions[index][2]) + 1)

                if instructions[index][0] == 'nop':
                    index += 1
                elif instructions[index][0] == 'acc':
                    acc += int(instructions[index][1])
                    index += 1
                elif instructions[index][0] == 'jmp':
                    index += int(instructions[index][1])

            if not_loop:
                not_solved = False
            else:
                if instructions[i][0] == 'nop':
                    instructions[i][0] = 'jmp'
                else:
                    instructions[i][0] = 'nop'
                day8ResetCounter(instructions)
        else:
            break

    return acc


def day8ResetCounter(instructions):
    for i in instructions:
        i[2] = '1'


def day8():
    print('Day 8')

    array = []
    with open('2020\input8.txt') as file:
        for data in file:
            data = data.rstrip().split(' ')
            data.append('1')
            array.append(data)

    acc = 0
    index = 0
    executed_instruction = []
    while True:
        if array[index][2] == '2':
            break

        array[index][2] = str(int(array[index][2]) + 1)

        if array[index][0] == 'nop':
            executed_instruction.append(index)
            index += 1
        elif array[index][0] == 'acc':
            acc += int(array[index][1])
            index += 1
        elif array[index][0] == 'jmp':
            executed_instruction.append(index)
            index += int(array[index][1])

    day8ResetCounter(array)

    print(f'Part 1: {acc}')
    print(f'Part 2: {day8Helper(array, executed_instruction)}')


def day9():
    print('Day 9')

    array = []
    with open('2020\input9.txt') as file:
        for data in file:
            data = data.rstrip()
            data = int(data)
            array.append(data)

    # part 1
    preamble_length = 25
    index = preamble_length
    found = False
    part1 = 0
    while index < len(array):
        boundary = array[index - preamble_length:index]

        for i in boundary:
            for t in boundary[1:]:
                if i + t == array[index]:
                    found = True
                    break
        if found:
            found = False
            index += 1
        else:
            part1 = array[index]
            print(f'Part 1: {part1}')
            break

    # part 2
    index = 0
    found = False
    part2 = []
    while index < len(array):
        part2 = [array[index]]
        for i in array[index + 1:]:
            if sum(part2) + i == part1:
                found = True
                break
            elif sum(part2) + i > part1:
                break
            part2.append(i)

        if not found:
            index += 1
        else:
            break

    part2 = sorted(part2)
    print(f'Part 2: {part2[0] + part2[-1]}')


def day10Helper(array):
    if len(array) == 1:
        return 1

    return 1


def day10():
    print('Day 10')
    # part 1
    array = []
    with open('2020\input10.txt') as file:
        for data in file:
            data = data.rstrip()
            array.append(int(data))

    array = sorted(array)
    start = 0
    different_jolt_dict = {1: 0, 2: 0, 3: 0}

    for i in array:
        different_jolt_dict[i - start] += 1
        start = i

    different_jolt_dict[3] += 1

    print(f'Part 1: {different_jolt_dict[1] * different_jolt_dict[3]}')

    # part 2 credit to https://www.reddit.com/user/kaur_virunurm/
    """
        - start from wall adapter (root node) with input count 1
        - add this count to the next 1, 2 or 3 adapters / nodes
        - add their input counts to next adapters / nodes
        - repeat this for all adapters (in sorted order)
        - you'll end up with input count for your device adapter
    """
    array.insert(0, 0)
    c = Counter({0: 1})
    for x in array:
        c[x + 1] += c[x]
        c[x + 2] += c[x]
        c[x + 3] += c[x]
    print(f'Part 2: {c[max(array) + 3]}')


def day11Helper(seats_map, seat_index, row_index, map_length, row_length):
    adjacent = []
    # first row
    if row_index == 0:
        # down
        adjacent.append(seats_map[row_index + 1][seat_index])
        if seat_index == 0:
            # right
            adjacent.append(seats_map[row_index][seat_index + 1])
            # down right
            adjacent.append(seats_map[row_index + 1][seat_index + 1])
            return adjacent
        elif seat_index == row_length - 1:
            # left
            adjacent.append(seats_map[row_index][seat_index - 1])
            # down left
            adjacent.append(seats_map[row_index + 1][seat_index - 1])
            return adjacent
        # right
        adjacent.append(seats_map[row_index][seat_index + 1])
        # down right
        adjacent.append(seats_map[row_index + 1][seat_index + 1])
        # left
        adjacent.append(seats_map[row_index][seat_index - 1])
        # down left
        adjacent.append(seats_map[row_index + 1][seat_index - 1])
        return adjacent
    # last row
    if row_index == map_length - 1:
        # up
        adjacent.append(seats_map[row_index - 1][seat_index])
        if seat_index == 0:
            # right
            adjacent.append(seats_map[row_index][seat_index + 1])
            # up right
            adjacent.append(seats_map[row_index - 1][seat_index + 1])
            return adjacent
        elif seat_index == row_length - 1:
            # left
            adjacent.append(seats_map[row_index][seat_index - 1])
            # up left
            adjacent.append(seats_map[row_index - 1][seat_index - 1])
            return adjacent
        # right
        adjacent.append(seats_map[row_index][seat_index + 1])
        # up right
        adjacent.append(seats_map[row_index - 1][seat_index + 1])
        # left
        adjacent.append(seats_map[row_index][seat_index - 1])
        # up left
        adjacent.append(seats_map[row_index - 1][seat_index - 1])
        return adjacent
    # other rows
    # up
    adjacent.append(seats_map[row_index - 1][seat_index])
    # down
    adjacent.append(seats_map[row_index + 1][seat_index])
    if seat_index == 0:
        # right
        adjacent.append(seats_map[row_index][seat_index + 1])
        # down right
        adjacent.append(seats_map[row_index + 1][seat_index + 1])
        # up right
        adjacent.append(seats_map[row_index - 1][seat_index + 1])
        return adjacent
    elif seat_index == row_length - 1:
        # left
        adjacent.append(seats_map[row_index][seat_index - 1])
        # down left
        adjacent.append(seats_map[row_index + 1][seat_index - 1])
        # up left
        adjacent.append(seats_map[row_index - 1][seat_index - 1])
        return adjacent
    # right
    adjacent.append(seats_map[row_index][seat_index + 1])
    # up right
    adjacent.append(seats_map[row_index - 1][seat_index + 1])
    # down right
    adjacent.append(seats_map[row_index + 1][seat_index + 1])
    # left
    adjacent.append(seats_map[row_index][seat_index - 1])
    # down left
    adjacent.append(seats_map[row_index + 1][seat_index - 1])
    # up left
    adjacent.append(seats_map[row_index - 1][seat_index - 1])
    return adjacent


def day11():
    print('Day 11')

    seats_map = []
    row_length = 0
    with open('2020\input11.txt') as file:
        for data in file:
            data = list(data.rstrip())
            if not row_length:
                row_length = len(data)
            seats_map.append(data)

    # part 1
    temp = []
    map_length = len(seats_map)
    count = 0
    while temp != seats_map:
        temp = copy.deepcopy(seats_map)

        for row_index, row in enumerate(temp):
            for seat_index, seat in enumerate(row):
                if seat == '.':
                    continue
                adjacent = day11Helper(temp, seat_index, row_index, map_length, row_length)

                if seat == 'L':
                    if not adjacent.count('#'):
                        seats_map[row_index][seat_index] = '#'
                elif seat == '#':
                    if adjacent.count('#') >= 4:
                        seats_map[row_index][seat_index] = 'L'
        count += 1

    part1 = 0
    for row in seats_map:
        for seat in row:
            if seat == '#':
                part1 += 1
    print(f'Part 1: {part1}')

    # TODO: part 2


def changeDirection(action, value, current_direction):
    direction_dict = {'E': 0, 'N': 90, 'W': 180, 'S': 270}
    if action == 'R':
        current_direction_value = direction_dict[current_direction]
        current_direction_value -= value
    else:
        current_direction_value = direction_dict[current_direction]
        current_direction_value += value

    if current_direction_value < 0:
        current_direction_value += 360
    elif current_direction_value >= 360:
        current_direction_value -= 360

    for i in direction_dict:
        if direction_dict[i] == current_direction_value:
            return i


def day12():
    print('Day 12')

    instructions = []
    with open('2020\input12.txt') as file:
        for data in file:
            data = data.rstrip()
            action = data[0]
            value = int(data[1:])
            instructions.append((action, value))

    direction = {'E': 0, 'W': 0, 'N': 0, 'S': 0}
    current_direction = 'E'

    for instruction in instructions:
        action = instruction[0]
        value = instruction[1]
        if action == 'R' or action == 'L':
            # print(current_direction, action, value, end=' ')
            current_direction = changeDirection(action, value, current_direction)
            # print(current_direction)
        elif action == 'F' or action == current_direction:
            direction[current_direction] += value
        elif action != current_direction:
            direction[action] += value

    print(direction)
    part1 = abs(direction['E'] - direction['W']) + abs(direction['N'] - direction['S'])
    print(f'Part 1: {part1}')

    # TODO: part 2


def chinese_remainder(n, a):
    temp_sum = 0
    prod = reduce(lambda a, b: a * b, n)
    for n_i, a_i in zip(n, a):
        p = prod // n_i
        temp_sum += a_i * mul_inv(p, n_i) * p
    return temp_sum % prod


def mul_inv(a, b):
    b0 = b
    x0, x1 = 0, 1
    if b == 1:
        return 1
    while a > 1:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += b0
    return x1


def day13():
    print('Day 13')
    time_stamp = 0
    buses = []
    buses2 = []
    with open('2020\input13.txt') as file:
        for index, data in enumerate(file):
            data = data.rstrip()
            if not index:
                time_stamp = int(data.rstrip())
            else:
                data = data.split(',')
                for i in data:
                    if i != 'x':
                        buses.append(int(i))
                        buses2.append(int(i))
                    else:
                        buses2.append(i)

    # part 1
    depart_time = []
    for i in buses:
        depart_time.append(time_stamp // i * i)
    for bus_num, value in enumerate(depart_time):
        while value < time_stamp:
            value += buses[bus_num]
        depart_time[bus_num] = value
    min_time = min(depart_time)
    print(f'Part 1: {(min_time - time_stamp) * buses[depart_time.index(min_time)]}')

    # part 2 Chinese remainder theorem
    n = []
    a = []
    for index, value in enumerate(buses2):
        if value != 'x':
            n.append(value)
            a.append(0 - index)
    print(n)
    print(a)
    print(f'Part 2: {chinese_remainder(n, a)}')


def toBinary(value):
    value = '{0:036b}'.format(int(value))
    return value


def day14Helper(mask, value):
    value = list(value)
    for index, i in enumerate(mask):
        if i != 'X':
            value[index] = i
    value = ''.join(value)
    return int(value, 2)


def day14HelperPart2(mask, value):
    x_indexes = []
    value = list(value)
    for index, i in enumerate(mask):
        if i == '0':
            continue
        elif i == 'X':
            value[index] = '0'
            x_indexes.append(35 - index)
        else:
            value[index] = i

    add = []
    for i in x_indexes:
        add.append(2 ** i)

    value = ''.join(value)
    value = int(value, 2)
    addresses = set()

    for index, i in enumerate(add):
        addresses.add(value + i)
        for t in add[index + 1:]:
            addresses.add(value + t + i)
    addresses.add(value)
    addresses.add(value + sum(add))
    return addresses


def day14():
    print('Day 14')
    mask = 0
    memory = {}
    # part 1
    with open('2020\input14.txt') as file:
        for data in file:
            data = data.rstrip()
            data = data.replace(' ', '').split('=')
            if data[0] == 'mask':
                mask = data[1]
            else:
                value = toBinary(data[1])
                memory_value = ''.join(c for c in data[0] if c.isdigit())
                memory[memory_value] = day14Helper(mask, value)
    part1 = 0
    for i in memory:
        part1 += memory[i]
    print(f'Part 1: {part1}')

    # TODO: part 2
    mask = 0
    memory = {}
    # with open('2020\input14.txt') as file:
    with open('2020\\test.txt') as file:
        for data in file:
            data = data.rstrip()
            data = data.replace(' ', '').split('=')
            if data[0] == 'mask':
                mask = data[1]
            else:
                value = int(data[1])
                memory_value = toBinary(''.join(c for c in data[0] if c.isdigit()))
                memory_value = day14HelperPart2(mask, memory_value)
                for i in memory_value:
                    memory[i] = value
    part2 = 0
    for i in memory:
        part2 += memory[i]
    print(f'Part 2: {part2}')


def day15Helper(array, turn_limit):
    turn = 1
    last = 0
    count = {}
    last_time_dict = defaultdict(list)
    while turn <= turn_limit:
        if turn <= len(array):
            last = array[turn - 1]
        else:
            if count[last] == 1:
                last = 0
            else:
                last = abs(last_time_dict[last][0] - last_time_dict[last][1])
        if last not in count:
            count[last] = 1
        else:
            count[last] += 1
        current_turn = last_time_dict[last]
        if len(current_turn) < 2:
            last_time_dict[last].append(turn)
        else:
            min_index = current_turn.index(min(current_turn))
            last_time_dict[last][min_index] = turn
        turn += 1
    return last


def day15():
    print('Day 15')
    # part 1
    array = [5, 1, 9, 18, 13, 8, 0]
    part1 = day15Helper(array, 2020)
    print(f'Part 1: the 2020th number is {part1}')
    # part 2
    part2 = day15Helper(array, 30000000)
    print(f'Part 2: the 30000000th number is {part2}')


def day16():
    print('Day 16')
    range_array = []
    range_dict = defaultdict(list)
    pattern = re.compile('[0-9]+')
    invalid = []
    invalid_index = set()
    my_ticket = None
    array = []
    # part 1
    with open('2020\input16.txt') as file:
        for line, data in enumerate(file):
            data = data.rstrip()
            if not len(data):
                continue
            if line < 20:
                temp = pattern.findall(data)
                temp2 = 0
                while temp2 <= 2:
                    range_array.append((int(temp[temp2]), int(temp[temp2 + 1])))
                    range_dict[line].append((int(temp[temp2]), int(temp[temp2 + 1])))
                    temp2 += 2
            elif data == 'your ticket:' or data == 'nearby tickets:':
                continue
            else:
                if not my_ticket:
                    my_ticket = data.split(',')
                else:
                    current_ticket = data.split(',')
                    array.append(current_ticket)
                    for index, value in enumerate(current_ticket):
                        value = int(value)
                        temp = False
                        for i in range_array:
                            lower, upper = i
                            if lower <= value <= upper:
                                temp = True
                                break
                            else:
                                temp = False
                        if not temp:
                            invalid.append(value)
                            invalid_index.add(line - 25)
    print(f'Part 1: {sum(invalid)}')
    # TODO: part 2
    ticket_index = []
    category_num = len(range_dict)
    limit = len(array) - len(invalid_index)
    for i in range_dict:
        current_range = range_dict[i]
        lower1 = current_range[0][0]
        upper1 = current_range[0][1]
        lower2 = current_range[1][0]
        upper2 = current_range[1][1]
        t = 0
        while t < category_num:
            if t in ticket_index:
                t += 1
                continue
            count = 0
            for ti, ticket in enumerate(array):
                if ti in invalid_index:
                    continue
                value = int(ticket[t])
                if lower1 <= value <= upper1 or lower2 <= value <= upper2:
                    count += 1
                else:
                    break
            if count == limit:
                ticket_index.append(t)
            t += 1
    print(ticket_index)
    part2 = 1
    for index, i in enumerate(ticket_index):
        if i <= 5:
            print(my_ticket[index])
            part2 *= int(my_ticket[index])
    print(f'Part 2: {part2}')


# TODO: day 17


def checkParenthesis(data):
    pattern = re.compile('\([^()]+\)')
    temp = []
    match = pattern.finditer(data)
    match = [i for i in match]
    if len(match):
        for i in match:
            lower, upper = i.span()
            temp.append((lower, upper, day18Helper(data[lower:upper + 1])))
        data = list(data)
        for i in temp:
            lower, upper, value = i
            data[lower] = value
            _ = lower + 1
            while _ < upper:
                data[_] = ''
                _ += 1
        temp = ''
        for i in data:
            temp += str(i)
        return checkParenthesis(temp)
    else:
        return day18Helper(data)


def day18Helper(data):
    data = data.replace('(', '')
    data = data.replace(')', '')
    data = data.split(' ')
    temp, op = 0, ''
    total = 0
    for i in data:
        if i.isdigit():
            if not total:
                total = int(i)
            else:
                temp = int(i)
        else:
            op = i
        if temp and op:
            if op == '+':
                total = total + temp
            elif op == '-':
                total = total - temp
            elif op == '/':
                total = total / temp
            elif op == '*':
                total = total * temp
            temp, op = 0, ''
    return total


def day18():
    print('Day 18')
    # part 1
    part1 = []
    with open('2020\input18.txt') as file:
        for data in file:
            data = data.rstrip()
            part1.append(checkParenthesis(data))
    print(f'Part 1: {sum(part1)}')
    # TODO: part 2


def day19Helper(rules_dict, special_keys):
    first = rules_dict['0'][0]
    temp = False
    for i in first:
        if not i.isdigit():
            temp = True
        else:
            temp = False
            break
    if temp:
        return
    for i in rules_dict:
        if i in special_keys:
            continue
        rules = rules_dict[i]
        for rule in rules:
            special = True
            temp = ''
            for index, value in enumerate(rule):
                if value in special_keys:
                    special_rules = rules_dict[value]
                    for special_rule in special_rules:
                        temp += ''.join(special_rule)
                    rule[index] = temp
                    temp = ''
                else:
                    special = False
            if special:
                special_keys.add(i)
    day19Helper(rules_dict, special_keys)


def day19():
    print('Day 19')
    # part 1
    rules_dict = defaultdict(list)
    change = False
    array = []
    special_keys = set()
    # with open('2020\input19.txt') as file:
    with open('2020\\test.txt') as file:
        for data in file:
            data = data.rstrip()
            if not len(data):
                change = True
                continue
            if not change:
                data = data.split(':')
                key = data[0]
                value = data[1].split()
                temp = []
                for i in value:
                    if i == '|':
                        rules_dict[key].append(temp)
                        temp = []
                    else:
                        if not i.isdigit():
                            i = i[1]
                            special_keys.add(key)
                        temp.append(str(i))
                rules_dict[key].append(temp)
            else:
                array.append(data)
    day19Helper(rules_dict, special_keys)
    print(rules_dict)
    print(special_keys)

# TODO: day 20


def day21():
    print('Day 21')
    # part 1
    # with open('2020\input21.txt') as file:
    pattern = re.compile('[^, ()]+')
    ingredients_count = {}
    allergens = []
    ingredients = []
    with open('2020\\test.txt') as file:
        for data in file:
            data = data.rstrip()
            data = pattern.findall(data)
            temp = data.index('contains')
            current_ingredients = data[:temp]
            ingredients.append(set(current_ingredients))
            allergens.append(set(data[temp + 1:]))
            for i in current_ingredients:
                if i in ingredients_count:
                    ingredients_count[i] += 1
                else:
                    ingredients_count[i] = 1
    for index, ingredient in enumerate(ingredients):
        to_remove = set()
        for i in ingredients[index + 1:]:
            for t in ingredient:
                if t in i:
                    to_remove.add(t)
        print(to_remove)
    print(ingredients)
    print(allergens)


def day22Helper(deck):
    mult = 1
    out = 0
    for i in reversed(deck):
        out += i * mult
        mult += 1
    return out


def day22Recursive(deck1, deck2):
    while len(deck1) and len(deck2):
        card1 = deck1[0]
        card2 = deck2[0]
        deck1.pop(0)
        deck2.pop(0)
        if card1 > card2:
            deck1.append(card1)
            deck1.append(card2)
        elif card2 > card1:
            deck2.append(card2)
            deck2.append(card1)
    return deck1, deck2


def day22():
    print('Day 22')
    # part 1
    deck1 = []
    deck2 = []
    flag_2 = False
    # with open('2020\input22.txt') as file:
    with open('2020\\test.txt') as file:
        for data in file:
            data = data.rstrip()
            if data == 'Player 1:' or not data:
                continue
            elif data == 'Player 2:':
                flag_2 = True
                continue
            if not flag_2:
                deck1.append(int(data))
            else:
                deck2.append(int(data))
    while len(deck1) and len(deck2):
        card1 = deck1[0]
        card2 = deck2[0]
        deck1.pop(0)
        deck2.pop(0)
        if card1 > card2:
            deck1.append(card1)
            deck1.append(card2)
        elif card2 > card1:
            deck2.append(card2)
            deck2.append(card1)
    if len(deck1):
        part1 = day22Helper(deck1)
    else:
        part1 = day22Helper(deck2)
    print(f'Part 1: {part1}')
    # part 2
    deck1, deck2 = day22Recursive(deck1, deck2)
    if len(deck1):
        part2 = day22Helper(deck1)
    else:
        part2 = day22Helper(deck2)
    print(f'Part 2: {part2}')


if __name__ == '__main__':
    day22()
