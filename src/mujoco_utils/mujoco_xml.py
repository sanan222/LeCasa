"""
Standalone MujocoXML class extracted from robosuite.
This provides XML manipulation functionality for MuJoCo models.
"""

import io
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET


def find_elements(root, tags=None, attribs=None, return_first=False):
    """
    Find elements in xml tree that match specified criteria.
    
    Args:
        root (ET.Element): Root element to search from
        tags (str or list): Tag names to search for
        attribs (dict): Attributes to match
        return_first (bool): If True, return only first match
    
    Returns:
        list or ET.Element: Found elements or first element if return_first=True
    """
    if tags is None:
        tags = []
    elif isinstance(tags, str):
        tags = [tags]
        
    if attribs is None:
        attribs = {}
    
    matches = []
    
    def _recursive_find(element):
        # Check if current element matches criteria
        if (not tags or element.tag in tags):
            match = True
            for attr, value in attribs.items():
                if element.get(attr) != value:
                    match = False
                    break
            if match:
                matches.append(element)
                if return_first:
                    return True
        
        # Search children
        for child in element:
            if _recursive_find(child):
                return True
        return False
    
    _recursive_find(root)
    
    if return_first:
        return matches[0] if matches else None
    return matches


class MujocoXML:
    """
    Base class of Mujoco xml file
    Wraps around ElementTree and provides additional functionality for merging different models.
    Specially, we keep track of <worldbody/>, <actuator/> and <asset/>

    When initialized, loads a mujoco xml from file.

    Args:
        fname (str): path to the MJCF xml file.
    """

    def __init__(self, fname):
        self.file = fname
        self.folder = os.path.dirname(fname)
        self.tree = ET.parse(fname)
        self.root = self.tree.getroot()
        self.worldbody = self.create_default_element("worldbody")
        self.actuator = self.create_default_element("actuator")
        self.sensor = self.create_default_element("sensor")
        self.asset = self.create_default_element("asset")
        self.tendon = self.create_default_element("tendon")
        self.equality = self.create_default_element("equality")
        self.contact = self.create_default_element("contact")

        # Skip default class processing to avoid XML schema violations
        # The original XML already has proper defaults applied
        pass

        self.resolve_asset_dependency()

    def create_default_element(self, name):
        """
        Create a default subelement if it does not exist.
        
        Args:
            name (str): name of the subelement
        
        Returns:
            ET.Element: the subelement
        """
        found = self.root.find(name)
        if found is not None:
            return found
        else:
            subelement = ET.SubElement(self.root, name)
            return subelement

    def resolve_asset_dependency(self):
        """
        Resolves relative paths in assets to absolute paths.
        """
        if self.folder == "":
            return

        for node in self.asset.findall(".//mesh[@file]"):
            file = node.get("file")
            if not os.path.isabs(file):
                abs_path = os.path.abspath(os.path.join(self.folder, file))
                node.set("file", abs_path)

        for node in self.asset.findall(".//texture[@file]"):
            file = node.get("file")
            if not os.path.isabs(file):
                abs_path = os.path.abspath(os.path.join(self.folder, file))
                node.set("file", abs_path)

    def merge(self, others, merge_body="default"):
        """
        Default merge method.

        Args:
            others (MujocoXML or list of MujocoXML): other xmls to merge into this one
                raises XML error if @others is not a MujocoXML instance.
                merges <worldbody/>, <actuator/> and <asset/> of @others into @self
            merge_body (None or str): If set, will merge child bodies of @others. Default is "default", which
                corresponds to the root worldbody for this XML. Otherwise, should be an existing body name
                that exists in this XML. None results in no merging of @other's bodies in its worldbody.
        """
        if type(others) is not list:
            others = [others]
        for idx, other in enumerate(others):
            if not isinstance(other, MujocoXML):
                raise ValueError(f"{type(other)} is not a MujocoXML instance.")
            if merge_body is not None:
                root = (
                    self.worldbody
                    if merge_body == "default"
                    else find_elements(
                        root=self.worldbody, tags="body", attribs={"name": merge_body}, return_first=True
                    )
                )
                for body in other.worldbody:
                    root.append(body)
            self.merge_assets(other)
            for one_actuator in other.actuator:
                self.actuator.append(one_actuator)
            for one_sensor in other.sensor:
                self.sensor.append(one_sensor)
            for one_tendon in other.tendon:
                self.tendon.append(one_tendon)
            for one_equality in other.equality:
                self.equality.append(one_equality)
            for one_contact in other.contact:
                self.contact.append(one_contact)

    def get_model(self, mode="mujoco"):
        """
        Generates a MjModel instance from the current xml tree.

        Args:
            mode (str): Mode with which to interpret xml tree

        Returns:
            MjModel: generated model from xml

        Raises:
            ValueError: [Invalid mode]
        """
        available_modes = ["mujoco"]
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            if mode == "mujoco":
                import mujoco
                model = mujoco.MjModel.from_xml_string(string.getvalue())
                return model
            raise ValueError("Unknown model mode: {}. Available options are: {}".format(mode, ",".join(available_modes)))

    def get_xml(self):
        """
        Reads a string of the MJCF XML file.

        Returns:
            str: XML tree read in from file
        """
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            return string.getvalue()

    def save_model(self, fname, pretty=False):
        """
        Saves the xml to file.

        Args:
            fname (str): output file location
            pretty (bool): If True, (attempts!! to) pretty print the output
        """
        with open(fname, "w") as f:
            xml_str = ET.tostring(self.root, encoding="unicode")
            if pretty:
                parsed_xml = xml.dom.minidom.parseString(xml_str)
                xml_str = parsed_xml.toprettyxml(newl="")
            f.write(xml_str)

    def merge_assets(self, other):
        """
        Merges @other's assets in a custom logic.

        Args:
            other (MujocoXML or MujocoObject): other xml file whose assets will be merged into this one
        """
        for asset in other.asset:
            if (
                find_elements(root=self.asset, tags=asset.tag, attribs={"name": asset.get("name")}, return_first=True)
                is None
            ):
                self.asset.append(asset)

    def get_element_names(self, root, element_type):
        """
        Searches recursively through the @root and returns a list of names of the specified @element_type

        Args:
            root (ET.Element): Root of the xml element tree to start recursively searching through
                (e.g.: `self.worldbody`)
            element_type (str): Name of element to return names of. (e.g.: "site", "geom", etc.)

        Returns:
            list: names that correspond to the specified @element_type
        """
        names = []
        for child in root:
            if child.tag == element_type:
                names.append(child.get("name"))
            names += self.get_element_names(child, element_type)
        return names

    @staticmethod
    def _get_default_classes(default):
        """
        Utility method to convert all default tags into a nested dictionary of values -- this will be used to replace
        all elements' class tags inline with the appropriate defaults if not specified.

        Args:
            default (ET.Element): Nested default tag XML root.

        Returns:
            dict: Nested dictionary, where each default class name is mapped to its own dict mapping element tag names
                (e.g.: geom, site, etc.) to the set of default attributes for that tag type
        """
        # Create nested dict to return
        default_dic = {}
        
        def _parse_defaults(elem):
            """Recursively parse nested default elements"""
            for child in elem:
                if child.tag == "default":
                    class_name = child.get("class")
                    if class_name:
                        # Create entry for this class
                        if class_name not in default_dic:
                            default_dic[class_name] = {}
                        # Add child elements as tag->element mappings
                        for grandchild in child:
                            if grandchild.tag != "default":  # Skip nested defaults
                                default_dic[class_name][grandchild.tag] = grandchild
                    # Recursively process nested defaults
                    _parse_defaults(child)
        
        _parse_defaults(default)
        return default_dic

    def _replace_defaults_inline(self, default_dic, root=None):
        """
        Utility method to replace all default class attributes recursively in the XML tree starting from @root
        with the corresponding defaults in @default_dic if they are not explicitly specified for ta given element.

        Args:
            root (ET.Element): Root of the xml element tree to start recursively replacing defaults. Only is used by
                recursive calls
            default_dic (dict): Nested dictionary, where each default class name is mapped to its own dict mapping
                element tag names (e.g.: geom, site, etc.) to the set of default attributes for that tag type
        """
        # If root is None, this is the top level call -- replace root with self.root
        if root is None:
            root = self.root
        # Check this current element if it contains any class elements
        cls_name = root.attrib.pop("class", None)
        if cls_name is not None and cls_name in default_dic:
            # If the tag for this element is contained in our default dic, we add any defaults that are not
            # explicitly specified in this
            tag_attrs = default_dic[cls_name].get(root.tag, None)
            if tag_attrs is not None:
                for k, v in tag_attrs.attrib.items():
                    if root.get(k, None) is None:
                        root.set(k, v)
        # Loop through all child elements
        for child in root:
            self._replace_defaults_inline(default_dic=default_dic, root=child)

    @property
    def name(self):
        """
        Returns name of this MujocoXML

        Returns:
            str: Name of this MujocoXML
        """
        return self.root.get("model") 