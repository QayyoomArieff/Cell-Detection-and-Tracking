import math

def Link(current, previous, id, tag, maxdist=100):

    # If no previous frame, then create new tree for each
    if not previous:
        for i in range (len(current)):
            current[i].append(str(tag))
            tag+=1

            id+=1
            current[i].append(str(id))

    else:
        pairs = {}
        cells = current.copy()

        while len(cells)>0:

            ignore = []
            notchild = True
            cell = cells.pop()

            while notchild:

                parent = nearest(cell, previous, maxdist, ignore)

                if parent:

                    parent = tuple(parent)

                    if parent not in pairs:
                        pairs[parent] = [cell]
                        break

                    else:
                        pairs[parent].append(cell)

                        if len(pairs[parent]) > 2:      # Parent only keeps the two closest centroid
                            notchild = furthest(parent, pairs[parent])
                            pairs[parent].remove(notchild)

                        else:
                            break

                else:
                    index1 = current.index(cell)
                    id += 1
                    current[index1].append(str(tag))
                    current[index1].append(str(id))
                    tag+=1
                    break

                if notchild != cell:
                    cell = notchild
                    ignore = []

                ignore.append(list(parent))


        # Check if parents have more than two centroids (Safety Check)
        for parent in list(pairs):
            if len(pairs[parent])>=2:

                cell1 = nearest(parent, pairs[parent], maxdist)
                index1 = current.index(cell1)
                current[index1].append(parent[2]+"."+str(1))
                pairs[parent].remove(cell1)

                id += 1
                current[index1].append(str(id))


                cell2 = nearest(parent, pairs[parent], maxdist)
                index2 = current.index(cell2)
                current[index2].append(parent[2] + "." + str(2))
                pairs[parent].remove(cell2)

                id += 1
                current[index2].append(str(id))

                for rest in pairs[parent]:
                    cells.append(rest)

                previous.remove(list(parent))
                pairs.pop(parent)

        for parent in pairs:
            cell = nearest(parent, pairs[parent], maxdist)

            index = current.index(cell)
            current[index].append(parent[2])
            current[index].append(parent[3])
            pairs[parent].remove(cell)

    return current, id


# Calculate the nearest centroid out of a list
def nearest(cell, cellset, maxdist, ignore=None):
    distance = maxdist
    nearest_cell = None

    for point in cellset:

        if ignore:
            if point in ignore:
                continue

        dist = math.sqrt((cell[0]-point[0])**(2)+(cell[1]-point[1])**2)

        if dist <= distance:
            distance = dist
            nearest_cell = point

    return nearest_cell


# Calculates the furthest centroid out of a list
def furthest(cell, cellset):
    distance = math.sqrt((cell[0]-cellset[0][0])**(2)+(cell[1]-cellset[0][1])**2)
    furthest_cell = cellset[0]

    for point in cellset:
        dist = math.sqrt((cell[0]-point[0])**(2)+(cell[1]-point[1])**2)

        if dist >= distance:
            distance = dist
            furthest_cell = point

    return furthest_cell

