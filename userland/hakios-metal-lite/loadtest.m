#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#include <stdio.h>
int main(void) {
	@autoreleasepool {
		NSBundle *b = [NSBundle bundleWithPath:@"/System/Library/Extensions/HakiosMetal.bundle"];
		printf("bundle: %s\n", b ? [[b bundlePath] UTF8String] : "nil");
		printf("before load: NSClassFromString = %p\n", (void *)NSClassFromString(@"HakiosMetalDevice"));
		NSError *e = nil;
		BOOL ok = [b loadAndReturnError:&e];
		printf("loadAndReturnError: %d %s\n", (int)ok, e ? [[e description] UTF8String] : "");
		Class c = NSClassFromString(@"HakiosMetalDevice");
		printf("after load: NSClassFromString = %p, objc_getClass = %p\n", (void *)c, (void *)objc_getClass("HakiosMetalDevice"));
		printf("classNamed: = %p\n", (void *)[b classNamed:@"HakiosMetalDevice"]);
		printf("principalClass = %p\n", (void *)[b principalClass]);
		if (c) {
			NSBundle *bc = [NSBundle bundleForClass:c];
			printf("bundleForClass = %s\n", bc ? [[bc bundlePath] UTF8String] : "nil");
			const char *img = class_getImageName(c);
			printf("class_getImageName = %s\n", img ? img : "(null)");
		}
	}
	return 0;
}
