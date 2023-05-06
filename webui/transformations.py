# We need this so Python doesn't complain about the unknown StableDiffusionProcessing-typehint at runtime
from __future__ import annotations

import csv
import os
import os.path
import typing
import collections.abc as abc
import tempfile
import shutil

#if typing.TYPE_CHECKING:
#    # Only import this when code is being type-checked, it doesn't have any effect at runtime
#    from .processing import StableDiffusionProcessing


class PromptTransformation(typing.NamedTuple):
    name: str
    regex: str
    replacement: str
    flags: str
    long_description: str


def merge_prompts(transformation_prompt: str, prompt: str) -> str:
    if "{prompt}" in transformation_prompt:
        res = transformation_prompt.replace("{prompt}", prompt)
    else:
        parts = filter(None, (prompt.strip(), transformation_prompt.strip()))
        res = ", ".join(parts)

    return res

import re
def apply_rule_to_prompt(rule, text):
    regex = rule.regex
    replacement = rule.replacement

    flags = 0
    if 'MULTILINE' in rule.flags:
        flags |= re.MULTILINE
    return re.sub(regex, replacement, text, flags=flags)

def apply_transformations_to_prompt(prompt, transformations):
    for transformation in transformations:
        #print(f"Applying transformation {transformation.name}, regex: {transformation.regex}, replacement: {transformation.replacement}, flags: {transformation.flags}")
        #print(f"Before: {prompt}") 
        prompt = apply_rule_to_prompt(transformation, prompt)

    return prompt


class TransformationDatabase:
    def __init__(self, path: str, user_path: str):
        self.no_transformation = PromptTransformation("None", "", "", "","")
        self.transformations = {}
        self.path = path
        self.user_path = user_path
        #print(path)
        self.reload()

    def reload(self):
        self.transformations.clear()

        if not os.path.exists(self.path):
            print(f"Can't find transformations at {self.path}")
        else:
            with open(self.path, "r", encoding="utf-8-sig", newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    #print(f"row: {row}")
                    # Support loading old CSV format with "name, text"-columns
                    regex = row["regex"]
                    replacement = row.get("replacement", "")
                    flags = row.get("flags", "")
                    long_description = row.get("long_description", "")
                    self.transformations[row["name"]] = PromptTransformation(row["name"], regex, replacement, flags, long_description)
    
        if not os.path.exists(self.user_path):
            print(f"Can't find transformations at {self.user_path}")
        else:
            with open(self.user_path, "r", encoding="utf-8-sig", newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    #print(f"row: {row}")
                    # Support loading old CSV format with "name, text"-columns
                    regex = row["regex"]
                    replacement = row.get("replacement", "")
                    flags = row.get("flags", "")
                    self.transformations[row["name"]] = PromptTransformation(row["name"], regex, replacement, flags, long_description)
    
    #def get_transformation_prompts(self, transformations):
    #    return [self.transformations.get(x, self.no_transformation).prompt for x in transformations]



    def apply_transformations_to_prompt(self, prompt, transformations):
        return apply_transformations_to_prompt(prompt, [self.transformations.get(x, self.no_transformation) for x in transformations])



    def save_transformations(self, path: str) -> None:
        # Always keep a backup file around
        if os.path.exists(path):
            shutil.copy(path, path + ".bak")

        fd = os.open(path, os.O_RDWR|os.O_CREAT)
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline='') as file:
            # _fields is actually part of the public API: typing.NamedTuple is a replacement for collections.NamedTuple,
            # and collections.NamedTuple has explicit documentation for accessing _fields. Same goes for _asdict()
            writer = csv.DictWriter(file, fieldnames=PromptTransformation._fields)
            writer.writeheader()
            writer.writerows(transformation._asdict() for k,     transformation in self.transformations.items())
